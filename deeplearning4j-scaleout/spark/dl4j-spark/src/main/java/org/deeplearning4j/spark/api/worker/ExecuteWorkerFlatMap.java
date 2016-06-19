package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.CommonSparkTrainingStats;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

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
        final boolean isGraph = dataConfig.isGraphNetwork();

        boolean stats = dataConfig.isCollectTrainingStats();
        StatsCalculationHelper s = (stats ? new StatsCalculationHelper() : null);
        if(stats) s.logMethodStartTime();

        if(!dataSetIterator.hasNext()){
            if(stats) s.logReturnTime();
            //TODO return the results...
            return Collections.emptyList();  //Sometimes: no data
        }

        int batchSize = dataConfig.getBatchSizePerWorker();
        final int prefetchCount = dataConfig.getPrefetchNumBatches();

        DataSetIterator batchedIterator = new IteratorDataSetIterator(dataSetIterator, batchSize);
        if(prefetchCount > 0){
            batchedIterator = new AsyncDataSetIterator(batchedIterator, prefetchCount);
        }

        try {
            MultiLayerNetwork net = null;
            ComputationGraph graph = null;
            if(stats) s.logInitialModelBefore();
            if(isGraph) graph = worker.getInitialModelGraph();
            else net = worker.getInitialModel();
            if(stats) s.logInitialModelAfter();

            int miniBatchCount = 0;
            int maxMinibatches = (dataConfig.getMaxBatchesPerWorker() > 0 ? dataConfig.getMaxBatchesPerWorker() : Integer.MAX_VALUE);

            while (batchedIterator.hasNext() && miniBatchCount++ < maxMinibatches) {
                if(stats) s.logNextDataSetBefore();
                DataSet next = batchedIterator.next();
                if(stats) s.logNextDataSetAfter(next.numExamples());

                if(stats){
                    s.logProcessMinibatchBefore();
                    Pair<R,SparkTrainingStats> result;
                    if(isGraph) result = worker.processMinibatchWithStats(next, graph, !batchedIterator.hasNext());
                    else result = worker.processMinibatchWithStats(next, net, !batchedIterator.hasNext());
                    s.logProcessMinibatchAfter();
                    if(result != null){
                        //Terminate training immediately
                        s.logReturnTime();
                        SparkTrainingStats workerStats = result.getSecond();
                        SparkTrainingStats returnStats = s.build(workerStats);
                        result.getFirst().setStats(returnStats);

                        return Collections.singletonList(result.getFirst());
                    }
                } else {
                    R result;
                    if(isGraph) result = worker.processMinibatch(next, graph, !batchedIterator.hasNext());
                    else result = worker.processMinibatch(next, net, !batchedIterator.hasNext());
                    if(result != null){
                        //Terminate training immediately
                        return Collections.singletonList(result);
                    }
                }
            }

            //For some reason, we didn't return already. Normally this shouldn't happen
            if(stats){
                s.logReturnTime();
                Pair<R,SparkTrainingStats> pair;
                if(isGraph) pair = worker.getFinalResultWithStats(graph);
                else pair = worker.getFinalResultWithStats(net);
                pair.getFirst().setStats(s.build(pair.getSecond()));
                return Collections.singletonList(pair.getFirst());
            } else {
                if(isGraph) return Collections.singletonList(worker.getFinalResult(graph));
                else return Collections.singletonList(worker.getFinalResult(net));
            }
        } finally {
            //Make sure we shut down the async thread properly...
            if(batchedIterator instanceof AsyncDataSetIterator){
                ((AsyncDataSetIterator)batchedIterator).shutdown();
            }
        }
    }
}

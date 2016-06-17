package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
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

        boolean stats = dataConfig.isCollectTrainingStats();
        StatsCalculationHelper s = (stats ? new StatsCalculationHelper() : null);
        if(stats) s.logMethodStartTime();


        if(!dataSetIterator.hasNext()){
            if(stats) s.logReturnTime();


            return Collections.emptyList();  //Sometimes: no data
        }

        int batchSize = dataConfig.getBatchSizePerWorker();
        final int prefetchCount = dataConfig.getPrefetchNumBatches();

        DataSetIterator batchedIterator = new IteratorDataSetIterator(dataSetIterator, batchSize);
        if(prefetchCount > 0){
            batchedIterator = new AsyncDataSetIterator(batchedIterator, prefetchCount);
        }

        try {
            if(stats) s.logInitialModelBefore();
            MultiLayerNetwork net = worker.getInitialModel();
            if(stats) s.logInitialModelAfter();

            int miniBatchCount = 0;
            int maxMinibatches = (dataConfig.getMaxBatchesPerWorker() > 0 ? dataConfig.getMaxBatchesPerWorker() : Integer.MAX_VALUE);

            while (batchedIterator.hasNext() && miniBatchCount++ < maxMinibatches) {
                System.out.println(Thread.currentThread().getId() + "\t" + miniBatchCount);
                if(stats) s.logNextDataSetBefore();
                DataSet next = batchedIterator.next();
                if(stats) s.logNextDataSetAfter(next.numExamples());

                if(stats){
                    s.logProcessMinibatchBefore();
                    Pair<R,SparkTrainingStats> result = worker.processMinibatchWithStats(next, net, batchedIterator.hasNext());
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
                    R result = worker.processMinibatch(next, net, batchedIterator.hasNext());
                    if(result != null){
                        //Terminate training immediately
                        return Collections.singletonList(result);
                    }
                }
            }

            //For some reason, we didn't return already. Normally this shouldn't happen
            if(stats){
                s.logReturnTime();
                Pair<R,SparkTrainingStats> pair = worker.getFinalResultWithStats(net);
                pair.getFirst().setStats(s.build(pair.getSecond()));
                return Collections.singletonList(pair.getFirst());
            } else {
                return Collections.singletonList(worker.getFinalResult(net));
            }
        } finally {
            //Make sure we shut down the async thread properly...
            if(batchedIterator instanceof AsyncDataSetIterator){
                ((AsyncDataSetIterator)batchedIterator).shutdown();
            }
        }
    }


    private static class StatsCalculationHelper {

        private long methodStartTime;
        private long returnTime;
        private long initalModelBefore;
        private long initialModelAfter;
        private long lastDataSetBefore;
        private long lastProcessBefore;
        private int totalExampleCount;
        //TODO: This adds more overhead than we want. Replace with a fast int collection (no boxing + conversion!)
        private List<Integer> dataSetGetTimes = new ArrayList<>();
        private List<Integer> processMiniBatchTimes = new ArrayList<>();

        private void logMethodStartTime(){
            methodStartTime = System.currentTimeMillis();
        }

        private void logReturnTime(){
            returnTime = System.currentTimeMillis();
        }

        private void logInitialModelBefore(){
            initalModelBefore = System.currentTimeMillis();
        }

        private void logInitialModelAfter(){
            initialModelAfter = System.currentTimeMillis();
        }

        private void logNextDataSetBefore(){
            lastDataSetBefore = System.currentTimeMillis();
        }

        private void logNextDataSetAfter(int numExamples){
            long now = System.currentTimeMillis();
            dataSetGetTimes.add((int)(now-lastDataSetBefore));
            totalExampleCount += numExamples;
        }

        private void logProcessMinibatchBefore(){
            lastProcessBefore = System.currentTimeMillis();
        }

        private void logProcessMinibatchAfter(){
            long now = System.currentTimeMillis();
            processMiniBatchTimes.add((int)(now-lastProcessBefore));
        }

        private CommonSparkTrainingStats build(SparkTrainingStats masterSpecificStats){
            //TODO again, do this without the lists...
            int[] dataSetGetTimesArr = new int[dataSetGetTimes.size()];
            for( int i=0; i<dataSetGetTimesArr.length; i++ ) dataSetGetTimesArr[i] = dataSetGetTimes.get(i);
            int[] processMiniBatchTimesArr = new int[processMiniBatchTimes.size()];
            for( int i=0; i<processMiniBatchTimesArr.length; i++ ) processMiniBatchTimesArr[i] = processMiniBatchTimes.get(i);

            return new CommonSparkTrainingStats.Builder()
                    .trainingMasterSpecificStats(masterSpecificStats)
                    .workerFlatMapTotalTimeMs((int)(returnTime-methodStartTime))
                    .workerFlatMapTotalExampleCount(totalExampleCount)
                    .workerFlatMapGetInitialModelTimeMs((int)(initialModelAfter-initalModelBefore))
                    .workerFlatMapDataSetGetTimesMs(dataSetGetTimesArr)
                    .workerFlatMapProcessMiniBatchTimesMs(processMiniBatchTimesArr)
                    .workerFlatMapCountNoDataInstances(dataSetGetTimes.size() == 0 ? 1 : 0)
                    .build();
        }
    }
}

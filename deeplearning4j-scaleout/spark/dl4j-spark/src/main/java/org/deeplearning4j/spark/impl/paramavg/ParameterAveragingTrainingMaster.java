package org.deeplearning4j.spark.impl.paramavg;

import lombok.Data;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerFlatMap;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerMultiDataSetFlatMap;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.graph.dataset.DataSetToMultiDataSetFn;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingAggregationTuple;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingElementAddFunction;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingElementCombineFunction;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.util.Iterator;

/**
 * ParameterAveragingTrainingMaster: A {@link TrainingMaster} implementation for spark-only training.
 * This is standard parameter averaging, using no
 */
@Data
public class ParameterAveragingTrainingMaster implements TrainingMaster<ParameterAveragingTrainingResult, ParameterAveragingTrainingWorker> {

    private static final Logger log = LoggerFactory.getLogger(ParameterAveragingTrainingMaster.class);

    private boolean saveUpdater;
    private int numWorkers;
    private int batchSizePerWorker;
    private int averagingFrequency;
    private int prefetchNumBatches;
    private boolean collectTrainingStats;
    private ParameterAveragingTrainingMasterStats.parameterAveragingTrainingMasterStatsHelper stats;


    private ParameterAveragingTrainingMaster(Builder builder){
        this.saveUpdater = builder.saveUpdater;
        this.numWorkers = builder.numWorkers;
        this.batchSizePerWorker = builder.batchSizePerWorker;
        this.averagingFrequency = builder.averagingFrequency;
        this.prefetchNumBatches = builder.prefetchNumBatches;
    }

    public ParameterAveragingTrainingMaster(boolean saveUpdater, int numWorkers, int batchSizePerWorker, int averagingFrequency, int prefetchNumBatches) {
        this(saveUpdater, numWorkers, batchSizePerWorker, averagingFrequency, prefetchNumBatches, false);
    }

    public ParameterAveragingTrainingMaster(boolean saveUpdater, int numWorkers, int batchSizePerWorker, int averagingFrequency, int prefetchNumBatches, boolean collectTrainingStats) {
        this.saveUpdater = saveUpdater;
        this.numWorkers = numWorkers;
        this.batchSizePerWorker = batchSizePerWorker;
        this.averagingFrequency = averagingFrequency;
        this.prefetchNumBatches = prefetchNumBatches;
        this.collectTrainingStats = collectTrainingStats;
        if(collectTrainingStats) stats = new ParameterAveragingTrainingMasterStats.parameterAveragingTrainingMasterStatsHelper();
    }

    @Override
    public ParameterAveragingTrainingWorker getWorkerInstance(SparkDl4jMultiLayer network) {
        NetBroadcastTuple tuple = new NetBroadcastTuple(network.getNetwork().getLayerWiseConfigurations(),
                network.getNetwork().params(),
                network.getNetwork().getUpdater());

        if(collectTrainingStats) stats.logBroadcastStart();
        Broadcast<NetBroadcastTuple> broadcast = network.getSparkContext().broadcast(tuple);
        if(collectTrainingStats) stats.logBroadcastEnd();

        WorkerConfiguration configuration = new WorkerConfiguration(false, batchSizePerWorker, averagingFrequency, prefetchNumBatches, collectTrainingStats);
        return new ParameterAveragingTrainingWorker(broadcast, saveUpdater, configuration);
    }

    @Override
    public ParameterAveragingTrainingWorker getWorkerInstance(SparkComputationGraph graph) {
        NetBroadcastTuple tuple = new NetBroadcastTuple(graph.getNetwork().getConfiguration(),
                graph.getNetwork().params(),
                graph.getNetwork().getUpdater());

        if(collectTrainingStats) stats.logBroadcastStart();
        Broadcast<NetBroadcastTuple> broadcast = graph.getSparkContext().broadcast(tuple);
        if(collectTrainingStats) stats.logBroadcastEnd();

        WorkerConfiguration configuration = new WorkerConfiguration(true, batchSizePerWorker, averagingFrequency, prefetchNumBatches, collectTrainingStats);
        return new ParameterAveragingTrainingWorker(broadcast, saveUpdater, configuration);
    }

    @Override
    public void executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        if(collectTrainingStats) stats.logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        trainingData.persist(StorageLevel.MEMORY_ONLY());

        //TODO: The assumption here is that each DataSet represents a single example. But this may not always be the case
        long totalCount = trainingData.count();
        int examplesPerSplit = numWorkers * batchSizePerWorker * averagingFrequency;

        JavaRDD<DataSet>[] splits;
        if(collectTrainingStats) stats.logSplitStart();
        if(totalCount <= examplesPerSplit){
            splits = (JavaRDD<DataSet>[])Array.newInstance(JavaRDD.class,1);
            splits[0] = trainingData;
        } else {
            int numSplits = (int)(totalCount/examplesPerSplit); //Intentional round down
            double[] weights = new double[numSplits];
            for( int i=0; i<weights.length; i++ ) weights[i] = 1.0 / numSplits;
            splits = trainingData.randomSplit(weights);
        }
        if(collectTrainingStats) stats.logSplitEnd();

        int splitNum = 1;
        for(JavaRDD<DataSet> split : splits) {
            log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, dataSetTotalExamples={}. Configured for {} executors",
                    splitNum, splits.length, batchSizePerWorker, averagingFrequency, totalCount, numWorkers);

            FlatMapFunction<Iterator<DataSet>, ParameterAveragingTrainingResult> function = new ExecuteWorkerFlatMap<>(getWorkerInstance(network));
            JavaRDD<ParameterAveragingTrainingResult> result = split.mapPartitions(function);
            processResults(network, null, result, splitNum, splits.length);

            splitNum++;
        }

        if(collectTrainingStats) stats.logFitEnd();
    }

    @Override
    public void executeTraining(SparkComputationGraph graph, JavaRDD<DataSet> trainingData){
        JavaRDD<MultiDataSet> mdsTrainingData = trainingData.map(new DataSetToMultiDataSetFn());

        executeTrainingMDS(graph, mdsTrainingData);
    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph graph, JavaRDD<MultiDataSet> trainingData){
        if(collectTrainingStats) stats.logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        trainingData.persist(StorageLevel.MEMORY_ONLY());

        //TODO: The assumption here is that each DataSet represents a single example. But this may not always be the case
        long totalCount = trainingData.count();
        int examplesPerSplit = numWorkers * batchSizePerWorker * averagingFrequency;

        JavaRDD<MultiDataSet>[] splits = randomSplit((int)totalCount, examplesPerSplit, trainingData);


        int splitNum = 1;
        for(JavaRDD<MultiDataSet> split : splits) {
            log.info("Starting graph training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, dataSetTotalExamples={}. Configured for {} executors",
                    splitNum, splits.length, batchSizePerWorker, averagingFrequency, totalCount, numWorkers);

            FlatMapFunction<Iterator<MultiDataSet>, ParameterAveragingTrainingResult> function = new ExecuteWorkerMultiDataSetFlatMap<>(getWorkerInstance(graph));
            JavaRDD<ParameterAveragingTrainingResult> result = split.mapPartitions(function);
            processResults(null, graph, result, splitNum, splits.length);

            splitNum++;
        }

        if(collectTrainingStats) stats.logFitEnd();
    }

    private <T> JavaRDD<T>[] randomSplit(int totalCount, int examplesPerSplit, JavaRDD<T> data){
        JavaRDD<T>[] splits;
        if(collectTrainingStats) stats.logSplitStart();
        if(totalCount <= examplesPerSplit){
            splits = (JavaRDD<T>[])Array.newInstance(JavaRDD.class,1);
            splits[0] = data;
        } else {
            int numSplits = totalCount/examplesPerSplit; //Intentional round down
            double[] weights = new double[numSplits];
            for( int i=0; i<weights.length; i++ ) weights[i] = 1.0 / numSplits;
            splits = data.randomSplit(weights);
        }
        if(collectTrainingStats) stats.logSplitEnd();
        return splits;
    }


    @Override
    public void setCollectTrainingStats(boolean collectTrainingStats) {
        this.collectTrainingStats = collectTrainingStats;
        if(collectTrainingStats){
            if(this.stats == null) this.stats = new ParameterAveragingTrainingMasterStats.parameterAveragingTrainingMasterStatsHelper();
        } else {
            this.stats = null;
        }
    }

    @Override
    public boolean getIsCollectTrainingStats() {
        return collectTrainingStats;
    }

    @Override
    public SparkTrainingStats getTrainingStats() {
        if(stats != null) return stats.build();
        return null;
    }


    private void processResults(SparkDl4jMultiLayer network, SparkComputationGraph graph, JavaRDD<ParameterAveragingTrainingResult> results, int splitNum, int totalSplits) {
        //Need to do parameter averaging, and where necessary also do averaging of the updaters

        //Let's do all of this in ONE step, such that we don't have extra synchronization costs

        if(collectTrainingStats) stats.logAggregateStartTime();
        ParameterAveragingAggregationTuple tuple = results.aggregate(null,
                new ParameterAveragingElementAddFunction(),
                new ParameterAveragingElementCombineFunction());
        INDArray params = tuple.getParametersSum();
        int aggCount = tuple.getAggregationsCount();
        SparkTrainingStats aggregatedStats = tuple.getSparkTrainingStats();
        if(collectTrainingStats) stats.logAggregationEndTime();


        if(collectTrainingStats) stats.logProcessParamsUpdaterStart();
        params.divi(aggCount);
        if(network != null){
            MultiLayerNetwork net = network.getNetwork();
            UpdaterAggregator updaterAg = tuple.getUpdaterAggregator();
            Updater updater = (updaterAg != null ? updaterAg.getUpdater() : null);
            net.setParameters(params);
            net.setUpdater(updater);

            network.setScore(tuple.getScoreSum() / tuple.getAggregationsCount());
        } else {
            ComputationGraph g = graph.getNetwork();
            ComputationGraphUpdater.Aggregator updaterAg = tuple.getUpdaterAggregatorGraph();
            ComputationGraphUpdater updater = (updaterAg != null ? updaterAg.getUpdater() : null);
            g.setParams(params);
            g.setUpdater(updater);

            graph.setScore(tuple.getScoreSum() / tuple.getAggregationsCount());
        }

        if(collectTrainingStats){
            stats.logProcessParamsUpdaterEnd();
            stats.addWorkerStats(aggregatedStats);
        }

        log.info("Completed training of split {} of {}", splitNum, totalSplits);
    }


    public static class Builder {

        private boolean saveUpdater;
        private int numWorkers;
        private int batchSizePerWorker = 16;
        private int averagingFrequency = 5;
        private int prefetchNumBatches = 0;

        /**
         * Create a builder, where the following number of workers (Spark executors) are used.
         * Note: this should match the
         *
         * @param numWorkers    Number of Spark executors in the cluster
         */
        public Builder(int numWorkers){
            this.numWorkers = numWorkers;
        }

        /**
         * Batch size per worker
         *
         * @param batchSizePerWorker    Size of each minibatch to use for each worker
         * @return
         */
        public Builder batchSizePerWorker(int batchSizePerWorker){
            this.batchSizePerWorker = batchSizePerWorker;
            return this;
        }

        /**
         * Frequency with which to average worker parameters.<br>
         * <b>Note</b>: Too high or too low can be bad for different reasons.<br>
         * - Too low (such as 1) can result in a lot of network traffic<br>
         * - Too high (>> 20 or so) can result in accuracy issues or problems with network convergence
         *
         * @param averagingFrequency    Frequency (in number of minibatches of size 'batchSizePerWorker') to average parameters
         */
        public Builder averagingFrequency(int averagingFrequency){
            if(averagingFrequency <= 0) throw new IllegalArgumentException("Ivalid input: averaging frequency must be >= 1");
            this.averagingFrequency = averagingFrequency;
            return this;
        }

        /**
         * Set the number of minibatches to asynchronously prefetch in the worker.
         *
         * Default: 0 (no prefetching)
         *
         * @param prefetchNumBatches    Number of minibatches (DataSets of size batchSizePerWorker) to fetch
         */
        public Builder workerPrefetchNumBatches(int prefetchNumBatches){
            this.prefetchNumBatches = prefetchNumBatches;
            return this;
        }

        /**
         * Set whether the updater (i.e., historical state for momentum, adagrad, etc should be saved).
         * <b>NOTE</b>: This can <b>double</b> (or more) the amount of network traffic in each direction, but might
         * improve network training performance (and can be more stable for certain updaters such as adagrad).<br>
         *
         * This is <b>enabled</b> by default.
         *
         * @param saveUpdater    If true: retain the updater state (default). If false, don't retain (updaters will be
         *                       reinitalized in each worker after averaging).
         */
        public Builder saveUpdater(boolean saveUpdater){
            this.saveUpdater = saveUpdater;
            return this;
        }

        public ParameterAveragingTrainingMaster build(){
            return new ParameterAveragingTrainingMaster(this);
        }
    }





}

package org.deeplearning4j.spark.impl.vanilla;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerFlatMap;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.vanilla.aggregator.VanillaAggregationTuple;
import org.deeplearning4j.spark.impl.vanilla.aggregator.VanillaElementAddFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.util.Iterator;

/**
 * Created by Alex on 14/06/2016.
 */
@AllArgsConstructor
public class VanillaTrainingMaster implements TrainingMaster<VanillaTrainingResult, VanillaTrainingWorker> {

    private static final Logger log = LoggerFactory.getLogger(VanillaTrainingMaster.class);

    //TODO do this configuration proprely
    private boolean saveUpdater;
    private int numWorkers;
    private int batchSizePerWorker;
    private int averagingFrequency;
    private int prefetchNumBatches;


    @Override
    public VanillaTrainingWorker getWorkerInstance(SparkDl4jMultiLayer network) {
        NetBroadcastTuple tuple = new NetBroadcastTuple(network.getNetwork().getLayerWiseConfigurations(),
                network.getNetwork().params(),
                network.getNetwork().getUpdater());

        Broadcast<NetBroadcastTuple> broadcast = network.getSparkContext().broadcast(tuple);

        WorkerConfiguration configuration = new WorkerConfiguration(batchSizePerWorker, averagingFrequency, prefetchNumBatches);
        return new VanillaTrainingWorker(broadcast, saveUpdater, configuration);
    }

    @Override
    public void executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {

        //For vanilla training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        trainingData.persist(StorageLevel.MEMORY_ONLY());

        //TODO: The assumption here is that each DataSet represents a single example. But this may not always be the case
        long totalCount = trainingData.count();
        int examplesPerSplit = numWorkers * batchSizePerWorker * averagingFrequency;

        JavaRDD<DataSet>[] splits;
        if(totalCount <= examplesPerSplit){
            splits = (JavaRDD<DataSet>[])new Object[]{trainingData};
        } else {
            int numSplits = (int)(totalCount/examplesPerSplit); //Intentional round down
            double[] weights = new double[numSplits];
            for( int i=0; i<weights.length; i++ ) weights[i] = 1.0 / numSplits;
            splits = trainingData.randomSplit(weights);
        }

        int splitNum = 1;
        for(JavaRDD<DataSet> split : splits) {
            log.info("Starting training of split {} of {}. MiniBatch={}, avgFrequency={}, totalExamples={}", splitNum, splits.length,
                    batchSizePerWorker, averagingFrequency, totalCount);

            FlatMapFunction<Iterator<DataSet>, VanillaTrainingResult> function = new ExecuteWorkerFlatMap<>(getWorkerInstance(network));
            JavaRDD<VanillaTrainingResult> result = split.mapPartitions(function);
            processResults(network, result);

            splitNum++;
        }


    }


    private void processResults(SparkDl4jMultiLayer network, JavaRDD<VanillaTrainingResult> results) {
        //Need to do parameter averaging, and where necessary also do averaging of the updaters

        //Let's do all of this in ONE step, such that we don't have extra synchronization costs

        VanillaAggregationTuple tuple = results.aggregate(null,
                new VanillaElementAddFunction(),

                );

    }
}

package org.deeplearning4j.spark.impl.paramavg;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 18/06/2016.
 */
public class TestCompareParameterAveragingSparkVsSingleMachine {

    private static MultiLayerConfiguration getConf(int seed, Updater updater){
        Nd4j.getRandom().setSeed(seed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.5)
                .weightInit(WeightInit.XAVIER)
                .updater(updater)
                .iterations(1)
                .seed(seed)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(10).nOut(10).build())
                .pretrain(false).backprop(true)
                .build();
        return conf;
    }

    private static JavaSparkContext getContext(int nWorkers){
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[" + nWorkers + "]");
        sparkConf.setAppName("Test");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        return sc;
    }

    private List<DataSet> getOneDataSetAsIndividalExamples(int totalExamples, int seed){
        Nd4j.getRandom().setSeed(seed);
        List<DataSet> list = new ArrayList<>();
        for( int i=0; i<totalExamples; i++ ){
            INDArray f = Nd4j.rand(1,10);
            INDArray l = Nd4j.rand(1,10);
            DataSet ds = new DataSet(f,l);
            list.add(ds);
        }
        return list;
    }

    private DataSet getOneDataSet(int totalExamples, int seed){
        return DataSet.merge(getOneDataSetAsIndividalExamples(totalExamples, seed));
    }

    @Test
    public void testOneExecutor(){
        //Idea: single worker/executor on Spark should give identical results to a single machine

        int miniBatchSize = 10;
        int nWorkers = 1;
        JavaSparkContext sc = getContext(nWorkers);

        try {

            ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(nWorkers)
                    .averagingFrequency(1)
                    .batchSizePerWorker(miniBatchSize)
                    .saveUpdater(true)
                    .workerPrefetchNumBatches(0)
                    .build();


            //Do training locally, for 3 minibatches
            int[] seeds = {1, 2, 3};

            MultiLayerNetwork net = new MultiLayerNetwork(getConf(12345, Updater.RMSPROP));
            net.init();
            INDArray initialParams = net.params().dup();

            for (int i = 0; i < seeds.length; i++) {
                DataSet ds = getOneDataSet(miniBatchSize, seeds[i]);
                net.fit(ds);
            }
            INDArray finalParams = net.params().dup();

            //Do training on Spark with one executor, for 3 separate minibatches
            SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, getConf(12345, Updater.RMSPROP), tm);
            sparkNet.setCollectTrainingStats(true);
            INDArray initialSparkParams = sparkNet.getNetwork().params().dup();

            for (int i = 0; i < seeds.length; i++) {
                List<DataSet> list = getOneDataSetAsIndividalExamples(miniBatchSize, seeds[i]);
                JavaRDD<DataSet> rdd = sc.parallelize(list);

                sparkNet.fit(rdd);
            }

            INDArray finalSparkParams = sparkNet.getNetwork().params().dup();

            assertEquals(initialParams, initialSparkParams);
            assertNotEquals(initialParams, finalParams);
            assertEquals(finalParams, finalSparkParams);
        } finally {
            sc.stop();
        }
    }

    @Test
    public void testAverageEveryStep(){
        //Idea: averaging every step with SGD (SGD updater + optimizer) is mathematically identical to doing the learning
        // on a single machine for synchronous distributed training
        //BUT: This is *ONLY* the case if all workers get an identical number of examples. This won't be the case if
        // we use RDD.randomSplit (which is what occurs if we use .fit(JavaRDD<DataSet> on a data set that needs splitting),
        // which might give a number of examples that isn't divisible by number of workers (like 39 examples on 4 executors)
        //This is also ONLY the case using SGD updater

        int miniBatchSizePerWorker = 10;
        int nWorkers = 4;
        JavaSparkContext sc = getContext(nWorkers);

        try {

            ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(nWorkers)
                    .averagingFrequency(1)
                    .batchSizePerWorker(miniBatchSizePerWorker)
                    .saveUpdater(true)
                    .workerPrefetchNumBatches(0)
                    .build();


            //Do training locally, for 3 minibatches
            int[] seeds = {1, 2, 3};

            MultiLayerNetwork net = new MultiLayerNetwork(getConf(12345, Updater.SGD));
            net.init();
            INDArray initialParams = net.params().dup();

            for (int i = 0; i < seeds.length; i++) {
                DataSet ds = getOneDataSet(miniBatchSizePerWorker * nWorkers, seeds[i]);
                net.fit(ds);
            }
            INDArray finalParams = net.params().dup();

            //Do training on Spark with one executor, for 3 separate minibatches
            SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, getConf(12345, Updater.SGD), tm);
            sparkNet.setCollectTrainingStats(true);
            INDArray initialSparkParams = sparkNet.getNetwork().params().dup();

            for (int i = 0; i < seeds.length; i++) {
                List<DataSet> list = getOneDataSetAsIndividalExamples(miniBatchSizePerWorker * nWorkers, seeds[i]);
                JavaRDD<DataSet> rdd = sc.parallelize(list);

                sparkNet.fit(rdd);
            }

            System.out.println(sparkNet.getSparkTrainingStats().statsAsString());

            INDArray finalSparkParams = sparkNet.getNetwork().params().dup();

            assertEquals(initialParams, initialSparkParams);
            assertNotEquals(initialParams, finalParams);
            assertEquals(finalParams, finalSparkParams);

            double sparkScore = sparkNet.getScore();
            assertTrue(sparkScore > 0.0);

            assertEquals(net.score(), sparkScore, 1e-3);
        } finally {
            sc.stop();
        }
    }
}

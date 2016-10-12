package org.deeplearning4j.spark.uistats;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 12/10/2016.
 */
public class TestListeners extends BaseSparkTest {

    @Test
    public void testStatsCollection(){

        JavaSparkContext sc = getContext();
        int nExecutors = numExecutors();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(10)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(100)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(100).nOut(3)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();



        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();


        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                .batchSizePerWorker(5)
                .averagingFrequency(3)
                .build();

        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc,conf,tm);
        StatsStorage ss = new MapDBStatsStorage();  //In-memory

        net.setListeners(ss, Collections.singletonList(new StatsListener(null)));

        List<DataSet> list = new IrisDataSetIterator(150,150).next().asList();
        JavaRDD<DataSet> rdd = sc.parallelize(list);

        net.fit(rdd);

        List<String> sessions = ss.listSessionIDs();
        System.out.println("Sessions: " + sessions);
        assertEquals(1, sessions.size());

        for(String s : sessions){
            List<String> typeIDs = ss.listTypeIDsForSession(s);
            List<String> workers = ss.listWorkerIDsForSession(s);
            System.out.println(s + "\t" + typeIDs + "\t" + workers);
        }
    }

}

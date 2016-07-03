package org.deeplearning4j.spark.canova;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 03/07/2016.
 */
public class TestPreProcessedData extends BaseSparkTest {

    @Test
    public void testPreprocessedData(){

        int dataSetObjSize = 5;
        int batchSizePerExecutor = 10;

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"dl4j_testpreprocdata");
        File f = new File(path);
        if(f.exists()) f.delete();
        f.mkdir();

        DataSetIterator iter = new IrisDataSetIterator(5,150);
        int i=0;
        while(iter.hasNext()){
            File f2 = new File(FilenameUtils.concat(path,"data" + (i++) + ".bin"));
            iter.next().save(f2);
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.RMSPROP)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .activation("tanh").build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(3)
                        .activation("softmax")
                        .build())
                .pretrain(false).backprop(true)
                .build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc,conf,
                new ParameterAveragingTrainingMaster.Builder(numExecutors(), dataSetObjSize)
                        .batchSizePerWorker(batchSizePerExecutor)
                        .averagingFrequency(1)
                        .repartionData(Repartition.Always)
                        .build());
        sparkNet.setCollectTrainingStats(true);

        JavaPairRDD<String,PortableDataStream> pds = sc.binaryFiles(path);
        assertEquals(150/dataSetObjSize, pds.count());

        sparkNet.fit(path);

        SparkTrainingStats sts = sparkNet.getSparkTrainingStats();
        int expNumFits = 12; //4 'fits' per averaging (4 executors, 1 averaging freq); 10 examples each -> 40 examples per fit. 150/40 = 3 averagings (round down); 3*4 = 12

        //Unfortunately: perfect partitioning isn't guaranteed by SparkUtils.balancedRandomSplit (esp. if original partitions are all size 1
        // which appears to be occurring at least some of the time), but we should get close to what we expect...
        assertTrue(Math.abs(expNumFits-sts.getValue("ParameterAveragingWorkerFitTimesMs").size()) < 3);

        assertEquals(3, sts.getValue("ParameterAveragingMasterMapPartitionsTimesMs").size());
    }

}

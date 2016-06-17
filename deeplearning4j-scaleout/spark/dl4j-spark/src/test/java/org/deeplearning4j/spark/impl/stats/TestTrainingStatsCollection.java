package org.deeplearning4j.spark.impl.stats;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.spark.api.stats.CommonSparkTrainingStats;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.vanilla.VanillaTrainingMaster;
import org.deeplearning4j.spark.impl.vanilla.stats.VanillaTrainingMasterStats;
import org.deeplearning4j.spark.impl.vanilla.stats.VanillaTrainingWorkerStats;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 17/06/2016.
 */
public class TestTrainingStatsCollection {

    @Test
    public void testStatsCollection() throws Exception {

        int nThreads = 4;

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[" + nThreads + "]");
        sparkConf.setAppName("Test");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).build())
                .pretrain(false).backprop(true)
                .build();

        int miniBatchSizePerWorker = 10;
        int averagingFrequency = 5;
        int numberOfAveragings = 3;

        int totalExamples = nThreads * miniBatchSizePerWorker * averagingFrequency * numberOfAveragings;

        Nd4j.getRandom().setSeed(12345);
        List<DataSet> list = new ArrayList<>();
        for( int i=0; i<totalExamples; i++ ){
            INDArray f = Nd4j.rand(1,10);
            INDArray l = Nd4j.rand(1,10);
            DataSet ds = new DataSet(f,l);
            list.add(ds);
        }

        JavaRDD<DataSet> rdd = sc.parallelize(list);
        rdd.repartition(4);

        VanillaTrainingMaster tm = new VanillaTrainingMaster.Builder(nThreads)
                .averagingFrequency(averagingFrequency)
                .batchSizePerWorker(miniBatchSizePerWorker)
                .saveUpdater(true)
                .workerPrefetchNumBatches(0)
                .build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);
        sparkNet.setCollectTrainingStats(true);
        sparkNet.fit(rdd);


        //Collect the expected keys:
        List<String> expectedStatNames = new ArrayList<>();
        Class<?>[] classes = new Class[]{CommonSparkTrainingStats.class, VanillaTrainingMasterStats.class, VanillaTrainingWorkerStats.class};
        String[] fieldNames = new String[]{"columnNames","columnNames","columnNames"};
        for(int i=0; i<classes.length; i++ ){
            Field field = classes[i].getDeclaredField(fieldNames[i]);
            field.setAccessible(true);
            Object f = field.get(null);
            Collection<String> c = (Collection<String>)f;
            expectedStatNames.addAll(c);
        }

        System.out.println(expectedStatNames);


        SparkTrainingStats stats = sparkNet.getSparkTrainingStats();
        Set<String> actualKeySet = stats.getKeySet();
        assertEquals(expectedStatNames.size(), actualKeySet.size());
        for(String s : stats.getKeySet()){
            assertTrue(expectedStatNames.contains(s));
            Object o = stats.getValue(s);
            System.out.println(s + "\t" + o);
        }

        System.out.println("\n\n\n");
        System.out.println(stats.statsAsString());


    }

}

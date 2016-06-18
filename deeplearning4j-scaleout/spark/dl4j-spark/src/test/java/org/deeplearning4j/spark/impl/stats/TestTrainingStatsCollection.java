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

import static junit.framework.TestCase.assertNotNull;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * Created by Alex on 17/06/2016.
 */
public class TestTrainingStatsCollection {

    @Test
    public void testStatsCollection() throws Exception {

        int nWorkers = 4;

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[" + nWorkers + "]");
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

        int totalExamples = nWorkers * miniBatchSizePerWorker * averagingFrequency * numberOfAveragings;

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

        VanillaTrainingMaster tm = new VanillaTrainingMaster.Builder(nWorkers)
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
        }

        String statsAsString = stats.statsAsString();
        System.out.println(statsAsString);
        assertEquals(actualKeySet.size(), statsAsString.split("\n").length);    //One line per stat


        //Go through nested stats
            //First: master stats
        assertTrue(stats instanceof VanillaTrainingMasterStats);
        VanillaTrainingMasterStats masterStats = (VanillaTrainingMasterStats)stats;
        int[] broadcastCreateTime = masterStats.getVanillaMasterBroadcastCreateTimesMs();
        assertEquals(numberOfAveragings, broadcastCreateTime.length);
        assertGreaterEqZero(broadcastCreateTime);

        int[] fitTimes = masterStats.getVanillaMasterFitTimesMs();
        assertEquals(1, fitTimes.length);   //i.e., number of times fit(JavaRDD<DataSet>) was called
        assertGreaterZero(fitTimes);

        int[] splitTimes = masterStats.getVanillaMasterSplitTimesMs();
        assertEquals(1, splitTimes.length);     //Splitting of the data set is executed once only (i.e., one fit(JavaRDD<DataSet>) call)
        assertGreaterEqZero(splitTimes);

        int[] aggregateTimesMs = masterStats.getVanillaMasterAggregateTimesMs();
        assertEquals(numberOfAveragings, aggregateTimesMs.length);
        assertGreaterEqZero(aggregateTimesMs);

        int[] processParamsTimesMs = masterStats.getVanillaMasterProcessParamsUpdaterTimesMs();
        assertEquals(numberOfAveragings, processParamsTimesMs.length);
        assertGreaterEqZero(processParamsTimesMs);

            //Second: Common spark training stats
        SparkTrainingStats commonStats = masterStats.getNestedTrainingStats();
        assertNotNull(commonStats);
        assertTrue(commonStats instanceof CommonSparkTrainingStats);
        CommonSparkTrainingStats cStats = (CommonSparkTrainingStats)commonStats;
        int[] workerFlatMapTotalTimeMs = cStats.getWorkerFlatMapTotalTimeMs();
        assertEquals(numberOfAveragings*nWorkers, workerFlatMapTotalTimeMs.length);
        assertGreaterZero(workerFlatMapTotalTimeMs);

        int[] workerFlatMapTotalExampleCount = cStats.getWorkerFlatMapTotalExampleCount();
        assertEquals(numberOfAveragings*nWorkers, workerFlatMapTotalExampleCount.length);
        assertGreaterZero(workerFlatMapTotalExampleCount);

        int[] workerFlatMapGetInitialModelTimeMs = cStats.getWorkerFlatMapGetInitialModelTimeMs();
        assertEquals(numberOfAveragings*nWorkers, workerFlatMapGetInitialModelTimeMs.length);
        assertGreaterEqZero(workerFlatMapGetInitialModelTimeMs);

        int[] workerFlatMapDataSetGetTimesMs = cStats.getWorkerFlatMapDataSetGetTimesMs();
        assertEquals(numberOfAveragings*nWorkers*averagingFrequency, workerFlatMapDataSetGetTimesMs.length);    //1 for every time we get a data set
        assertGreaterEqZero(workerFlatMapDataSetGetTimesMs);

        int[] workerFlatMapProcessMiniBatchTimesMs = cStats.getWorkerFlatMapProcessMiniBatchTimesMs();
        assertEquals(numberOfAveragings*nWorkers*averagingFrequency, workerFlatMapProcessMiniBatchTimesMs.length);
        assertGreaterEqZero(workerFlatMapProcessMiniBatchTimesMs);

        int workerFlatMapCountNoDataInstances = cStats.getWorkerFlatMapCountNoDataInstances();
        assertEquals(0, workerFlatMapCountNoDataInstances);

            //Third: VanillaTrainingWorker stats
        SparkTrainingStats vanillaStats = cStats.getNestedTrainingStats();
        assertNotNull(vanillaStats);
        assertTrue(vanillaStats instanceof VanillaTrainingWorkerStats);

        VanillaTrainingWorkerStats vStats = (VanillaTrainingWorkerStats)vanillaStats;
        int[] vanillaWorkerBroadcastGetValueTimeMs = vStats.getVanillaWorkerBroadcastGetValueTimeMs();
        assertEquals(numberOfAveragings*nWorkers, vanillaWorkerBroadcastGetValueTimeMs.length);
        assertGreaterEqZero(vanillaWorkerBroadcastGetValueTimeMs);

        int[] vanillaWorkerInitTimeMs = vStats.getVanillaWorkerInitTimeMs();
        assertEquals(numberOfAveragings*nWorkers, vanillaWorkerInitTimeMs.length);
        assertGreaterEqZero(vanillaWorkerInitTimeMs);

        int[] vanillaWorkerFitTimesMs = vStats.getVanillaWorkerFitTimesMs();
        assertEquals(numberOfAveragings*nWorkers*averagingFrequency, vanillaWorkerFitTimesMs.length);
        assertGreaterEqZero(vanillaWorkerFitTimesMs);

        assertNull(vStats.getNestedTrainingStats());
    }

    private static void assertGreaterEqZero(int[] array){
        for(int i : array) assertTrue(i >= 0);
    }

    private static void assertGreaterZero(int[] array){
        for(int i : array) assertTrue(i > 0);
    }
}

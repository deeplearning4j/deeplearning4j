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
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingWorkerStats;
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

        try {

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
            for (int i = 0; i < totalExamples; i++) {
                INDArray f = Nd4j.rand(1, 10);
                INDArray l = Nd4j.rand(1, 10);
                DataSet ds = new DataSet(f, l);
                list.add(ds);
            }

            JavaRDD<DataSet> rdd = sc.parallelize(list);
            rdd.repartition(4);

            ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(nWorkers)
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
            Class<?>[] classes = new Class[]{CommonSparkTrainingStats.class, ParameterAveragingTrainingMasterStats.class, ParameterAveragingTrainingWorkerStats.class};
            String[] fieldNames = new String[]{"columnNames", "columnNames", "columnNames"};
            for (int i = 0; i < classes.length; i++) {
                Field field = classes[i].getDeclaredField(fieldNames[i]);
                field.setAccessible(true);
                Object f = field.get(null);
                Collection<String> c = (Collection<String>) f;
                expectedStatNames.addAll(c);
            }

            System.out.println(expectedStatNames);


            SparkTrainingStats stats = sparkNet.getSparkTrainingStats();
            Set<String> actualKeySet = stats.getKeySet();
            assertEquals(expectedStatNames.size(), actualKeySet.size());
            for (String s : stats.getKeySet()) {
                assertTrue(expectedStatNames.contains(s));
                Object o = stats.getValue(s);
            }

            String statsAsString = stats.statsAsString();
            System.out.println(statsAsString);
            assertEquals(actualKeySet.size(), statsAsString.split("\n").length);    //One line per stat


            //Go through nested stats
            //First: master stats
            assertTrue(stats instanceof ParameterAveragingTrainingMasterStats);
            ParameterAveragingTrainingMasterStats masterStats = (ParameterAveragingTrainingMasterStats) stats;
            int[] broadcastCreateTime = masterStats.getParameterAveragingMasterBroadcastCreateTimesMs();
            assertEquals(numberOfAveragings, broadcastCreateTime.length);
            assertGreaterEqZero(broadcastCreateTime);

            int[] fitTimes = masterStats.getParameterAveragingMasterFitTimesMs();
            assertEquals(1, fitTimes.length);   //i.e., number of times fit(JavaRDD<DataSet>) was called
            assertGreaterZero(fitTimes);

            int[] splitTimes = masterStats.getParameterAveragingMasterSplitTimesMs();
            assertEquals(1, splitTimes.length);     //Splitting of the data set is executed once only (i.e., one fit(JavaRDD<DataSet>) call)
            assertGreaterEqZero(splitTimes);

            int[] aggregateTimesMs = masterStats.getParamaterAveragingMasterAggregateTimesMs();
            assertEquals(numberOfAveragings, aggregateTimesMs.length);
            assertGreaterEqZero(aggregateTimesMs);

            int[] processParamsTimesMs = masterStats.getParameterAveragingMasterProcessParamsUpdaterTimesMs();
            assertEquals(numberOfAveragings, processParamsTimesMs.length);
            assertGreaterEqZero(processParamsTimesMs);

            //Second: Common spark training stats
            SparkTrainingStats commonStats = masterStats.getNestedTrainingStats();
            assertNotNull(commonStats);
            assertTrue(commonStats instanceof CommonSparkTrainingStats);
            CommonSparkTrainingStats cStats = (CommonSparkTrainingStats) commonStats;
            int[] workerFlatMapTotalTimeMs = cStats.getWorkerFlatMapTotalTimeMs();
            assertEquals(numberOfAveragings * nWorkers, workerFlatMapTotalTimeMs.length);
            assertGreaterZero(workerFlatMapTotalTimeMs);

            int[] workerFlatMapTotalExampleCount = cStats.getWorkerFlatMapTotalExampleCount();
            assertEquals(numberOfAveragings * nWorkers, workerFlatMapTotalExampleCount.length);
            assertGreaterZero(workerFlatMapTotalExampleCount);

            int[] workerFlatMapGetInitialModelTimeMs = cStats.getWorkerFlatMapGetInitialModelTimeMs();
            assertEquals(numberOfAveragings * nWorkers, workerFlatMapGetInitialModelTimeMs.length);
            assertGreaterEqZero(workerFlatMapGetInitialModelTimeMs);

            int[] workerFlatMapDataSetGetTimesMs = cStats.getWorkerFlatMapDataSetGetTimesMs();
            int numMinibatchesProcessed = workerFlatMapDataSetGetTimesMs.length;
            int expectedNumMinibatchesProcessed = numberOfAveragings * nWorkers * averagingFrequency;   //1 for every time we get a data set

            //Sometimes random split is just bad - some executors might miss out on getting the expected amount of data
            assertTrue(numMinibatchesProcessed >= expectedNumMinibatchesProcessed - 5);

            int workerFlatMapCountNoDataInstances = cStats.getWorkerFlatMapCountNoDataInstances();
            if(numMinibatchesProcessed == expectedNumMinibatchesProcessed){
                assertEquals(0, workerFlatMapCountNoDataInstances);
            }
            assertGreaterEqZero(workerFlatMapDataSetGetTimesMs);

            int[] workerFlatMapProcessMiniBatchTimesMs = cStats.getWorkerFlatMapProcessMiniBatchTimesMs();
            assertTrue(workerFlatMapProcessMiniBatchTimesMs.length >= numberOfAveragings * nWorkers * averagingFrequency - 5 );
            assertGreaterEqZero(workerFlatMapProcessMiniBatchTimesMs);

            //Third: ParameterAveragingTrainingWorker stats
            SparkTrainingStats paramAvgStats = cStats.getNestedTrainingStats();
            assertNotNull(paramAvgStats);
            assertTrue(paramAvgStats instanceof ParameterAveragingTrainingWorkerStats);

            ParameterAveragingTrainingWorkerStats pStats = (ParameterAveragingTrainingWorkerStats) paramAvgStats;
            int[] parameterAveragingWorkerBroadcastGetValueTimeMs = pStats.getParameterAveragingWorkerBroadcastGetValueTimeMs();
            assertEquals(numberOfAveragings * nWorkers, parameterAveragingWorkerBroadcastGetValueTimeMs.length);
            assertGreaterEqZero(parameterAveragingWorkerBroadcastGetValueTimeMs);

            int[] parameterAveragingWorkerInitTimeMs = pStats.getParameterAveragingWorkerInitTimeMs();
            assertEquals(numberOfAveragings * nWorkers, parameterAveragingWorkerInitTimeMs.length);
            assertGreaterEqZero(parameterAveragingWorkerInitTimeMs);

            int[] parameterAveragingWorkerFitTimesMs = pStats.getParameterAveragingWorkerFitTimesMs();
            assertEquals(numberOfAveragings * nWorkers * averagingFrequency, parameterAveragingWorkerFitTimesMs.length);
            assertGreaterEqZero(parameterAveragingWorkerFitTimesMs);

            assertNull(pStats.getNestedTrainingStats());
        } finally {
            sc.stop();
        }
    }

    private static void assertGreaterEqZero(int[] array){
        for(int i : array) assertTrue(i >= 0);
    }

    private static void assertGreaterZero(int[] array){
        for(int i : array) assertTrue(i > 0);
    }
}

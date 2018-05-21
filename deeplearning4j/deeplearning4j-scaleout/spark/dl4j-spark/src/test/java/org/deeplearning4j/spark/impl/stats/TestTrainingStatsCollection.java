package org.deeplearning4j.spark.impl.stats;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.stats.CommonSparkTrainingStats;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingWorkerStats;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.util.*;

import static junit.framework.TestCase.assertNotNull;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.*;

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
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                            .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                            .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).build()).pretrain(false).backprop(true)
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

            ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(nWorkers, 1)
                            .averagingFrequency(averagingFrequency).batchSizePerWorker(miniBatchSizePerWorker)
                            .saveUpdater(true).workerPrefetchNumBatches(0).repartionData(Repartition.Always).build();

            SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);
            sparkNet.setCollectTrainingStats(true);
            sparkNet.fit(rdd);


            //Collect the expected keys:
            List<String> expectedStatNames = new ArrayList<>();
            Class<?>[] classes = new Class[] {CommonSparkTrainingStats.class,
                            ParameterAveragingTrainingMasterStats.class, ParameterAveragingTrainingWorkerStats.class};
            String[] fieldNames = new String[] {"columnNames", "columnNames", "columnNames"};
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
                assertNotNull(stats.getValue(s));
            }

            String statsAsString = stats.statsAsString();
            System.out.println(statsAsString);
            assertEquals(actualKeySet.size(), statsAsString.split("\n").length); //One line per stat


            //Go through nested stats
            //First: master stats
            assertTrue(stats instanceof ParameterAveragingTrainingMasterStats);
            ParameterAveragingTrainingMasterStats masterStats = (ParameterAveragingTrainingMasterStats) stats;

            List<EventStats> exportTimeStats = masterStats.getParameterAveragingMasterExportTimesMs();
            assertEquals(1, exportTimeStats.size());
            assertDurationGreaterZero(exportTimeStats);
            assertNonNullFields(exportTimeStats);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(exportTimeStats, 1, 1, 1);

            List<EventStats> countRddTime = masterStats.getParameterAveragingMasterCountRddSizeTimesMs();
            assertEquals(1, countRddTime.size()); //occurs once per fit
            assertDurationGreaterEqZero(countRddTime);
            assertNonNullFields(countRddTime);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(countRddTime, 1, 1, 1); //should occur only in master once

            List<EventStats> broadcastCreateTime = masterStats.getParameterAveragingMasterBroadcastCreateTimesMs();
            assertEquals(numberOfAveragings, broadcastCreateTime.size());
            assertDurationGreaterEqZero(broadcastCreateTime);
            assertNonNullFields(broadcastCreateTime);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(broadcastCreateTime, 1, 1, 1); //only 1 thread for master

            List<EventStats> fitTimes = masterStats.getParameterAveragingMasterFitTimesMs();
            assertEquals(1, fitTimes.size()); //i.e., number of times fit(JavaRDD<DataSet>) was called
            assertDurationGreaterZero(fitTimes);
            assertNonNullFields(fitTimes);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(fitTimes, 1, 1, 1); //only 1 thread for master

            List<EventStats> splitTimes = masterStats.getParameterAveragingMasterSplitTimesMs();
            assertEquals(1, splitTimes.size()); //Splitting of the data set is executed once only (i.e., one fit(JavaRDD<DataSet>) call)
            assertDurationGreaterEqZero(splitTimes);
            assertNonNullFields(splitTimes);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(splitTimes, 1, 1, 1); //only 1 thread for master

            List<EventStats> aggregateTimesMs = masterStats.getParamaterAveragingMasterAggregateTimesMs();
            assertEquals(numberOfAveragings, aggregateTimesMs.size());
            assertDurationGreaterEqZero(aggregateTimesMs);
            assertNonNullFields(aggregateTimesMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(aggregateTimesMs, 1, 1, 1); //only 1 thread for master

            List<EventStats> processParamsTimesMs =
                            masterStats.getParameterAveragingMasterProcessParamsUpdaterTimesMs();
            assertEquals(numberOfAveragings, processParamsTimesMs.size());
            assertDurationGreaterEqZero(processParamsTimesMs);
            assertNonNullFields(processParamsTimesMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(processParamsTimesMs, 1, 1, 1); //only 1 thread for master

            List<EventStats> repartitionTimesMs = masterStats.getParameterAveragingMasterRepartitionTimesMs();
            assertEquals(numberOfAveragings, repartitionTimesMs.size());
            assertDurationGreaterEqZero(repartitionTimesMs);
            assertNonNullFields(repartitionTimesMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(repartitionTimesMs, 1, 1, 1); //only 1 thread for master

            //Second: Common spark training stats
            SparkTrainingStats commonStats = masterStats.getNestedTrainingStats();
            assertNotNull(commonStats);
            assertTrue(commonStats instanceof CommonSparkTrainingStats);
            CommonSparkTrainingStats cStats = (CommonSparkTrainingStats) commonStats;
            List<EventStats> workerFlatMapTotalTimeMs = cStats.getWorkerFlatMapTotalTimeMs();
            assertEquals(numberOfAveragings * nWorkers, workerFlatMapTotalTimeMs.size());
            assertDurationGreaterZero(workerFlatMapTotalTimeMs);
            assertNonNullFields(workerFlatMapTotalTimeMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(workerFlatMapTotalTimeMs, 1, 1, nWorkers);

            List<EventStats> workerFlatMapGetInitialModelTimeMs = cStats.getWorkerFlatMapGetInitialModelTimeMs();
            assertEquals(numberOfAveragings * nWorkers, workerFlatMapGetInitialModelTimeMs.size());
            assertDurationGreaterEqZero(workerFlatMapGetInitialModelTimeMs);
            assertNonNullFields(workerFlatMapGetInitialModelTimeMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(workerFlatMapGetInitialModelTimeMs, 1, 1, nWorkers);

            List<EventStats> workerFlatMapDataSetGetTimesMs = cStats.getWorkerFlatMapDataSetGetTimesMs();
            int numMinibatchesProcessed = workerFlatMapDataSetGetTimesMs.size();
            int expectedNumMinibatchesProcessed = numberOfAveragings * nWorkers * averagingFrequency; //1 for every time we get a data set

            //Sometimes random split is just bad - some executors might miss out on getting the expected amount of data
            assertTrue(numMinibatchesProcessed >= expectedNumMinibatchesProcessed - 5);

            List<EventStats> workerFlatMapProcessMiniBatchTimesMs = cStats.getWorkerFlatMapProcessMiniBatchTimesMs();
            assertTrue(workerFlatMapProcessMiniBatchTimesMs.size() >= numberOfAveragings * nWorkers * averagingFrequency
                            - 5);
            assertDurationGreaterEqZero(workerFlatMapProcessMiniBatchTimesMs);
            assertNonNullFields(workerFlatMapDataSetGetTimesMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(workerFlatMapDataSetGetTimesMs, 1, 1, nWorkers);

            //Third: ParameterAveragingTrainingWorker stats
            SparkTrainingStats paramAvgStats = cStats.getNestedTrainingStats();
            assertNotNull(paramAvgStats);
            assertTrue(paramAvgStats instanceof ParameterAveragingTrainingWorkerStats);

            ParameterAveragingTrainingWorkerStats pStats = (ParameterAveragingTrainingWorkerStats) paramAvgStats;
            List<EventStats> parameterAveragingWorkerBroadcastGetValueTimeMs =
                            pStats.getParameterAveragingWorkerBroadcastGetValueTimeMs();
            assertEquals(numberOfAveragings * nWorkers, parameterAveragingWorkerBroadcastGetValueTimeMs.size());
            assertDurationGreaterEqZero(parameterAveragingWorkerBroadcastGetValueTimeMs);
            assertNonNullFields(parameterAveragingWorkerBroadcastGetValueTimeMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(parameterAveragingWorkerBroadcastGetValueTimeMs, 1, 1,
                            nWorkers);

            List<EventStats> parameterAveragingWorkerInitTimeMs = pStats.getParameterAveragingWorkerInitTimeMs();
            assertEquals(numberOfAveragings * nWorkers, parameterAveragingWorkerInitTimeMs.size());
            assertDurationGreaterEqZero(parameterAveragingWorkerInitTimeMs);
            assertNonNullFields(parameterAveragingWorkerInitTimeMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(parameterAveragingWorkerInitTimeMs, 1, 1, nWorkers);

            List<EventStats> parameterAveragingWorkerFitTimesMs = pStats.getParameterAveragingWorkerFitTimesMs();
            assertTrue(parameterAveragingWorkerFitTimesMs.size() >= numberOfAveragings * nWorkers * averagingFrequency
                            - 5);
            assertDurationGreaterEqZero(parameterAveragingWorkerFitTimesMs);
            assertNonNullFields(parameterAveragingWorkerFitTimesMs);
            assertExpectedNumberMachineIdsJvmIdsThreadIds(parameterAveragingWorkerFitTimesMs, 1, 1, nWorkers);

            assertNull(pStats.getNestedTrainingStats());


            //Finally: try exporting stats
            String tempDir = System.getProperty("java.io.tmpdir");
            String outDir = FilenameUtils.concat(tempDir, "dl4j_testTrainingStatsCollection");
            stats.exportStatFiles(outDir, sc.sc());

            String htmlPlotsPath = FilenameUtils.concat(outDir, "AnalysisPlots.html");
            StatsUtils.exportStatsAsHtml(stats, htmlPlotsPath, sc);

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            StatsUtils.exportStatsAsHTML(stats, baos);
            baos.close();
            byte[] bytes = baos.toByteArray();
            String str = new String(bytes, "UTF-8");
            //            System.out.println(str);
        } finally {
            sc.stop();
        }
    }

    private static void assertDurationGreaterEqZero(List<EventStats> array) {
        for (EventStats e : array)
            assertTrue(e.getDurationMs() >= 0);
    }

    private static void assertDurationGreaterZero(List<EventStats> array) {
        for (EventStats e : array)
            assertTrue(e.getDurationMs() > 0);
    }

    private static void assertNonNullFields(List<EventStats> array) {
        for (EventStats e : array) {
            assertNotNull(e.getMachineID());
            assertNotNull(e.getJvmID());
            assertNotNull(e.getDurationMs());
            assertFalse(e.getMachineID().isEmpty());
            assertFalse(e.getJvmID().isEmpty());
            assertTrue(e.getThreadID() > 0);
        }
    }

    private static void assertExpectedNumberMachineIdsJvmIdsThreadIds(List<EventStats> events, int expNMachineIDs,
                    int expNumJvmIds, int expNumThreadIds) {
        Set<String> machineIDs = new HashSet<>();
        Set<String> jvmIDs = new HashSet<>();
        Set<Long> threadIDs = new HashSet<>();
        for (EventStats e : events) {
            machineIDs.add(e.getMachineID());
            jvmIDs.add(e.getJvmID());
            threadIDs.add(e.getThreadID());
        }
        assertTrue(machineIDs.size() == expNMachineIDs);
        assertTrue(jvmIDs.size() == expNumJvmIds);
        assertTrue(threadIDs.size() == expNumThreadIds);
    }
}

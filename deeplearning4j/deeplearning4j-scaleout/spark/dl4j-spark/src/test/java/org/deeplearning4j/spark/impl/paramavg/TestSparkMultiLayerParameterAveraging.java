/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.impl.paramavg;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.ExampleCountEventStats;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.Assert.*;


/**
 * Created by agibsonccc on 1/18/15.
 */
public class TestSparkMultiLayerParameterAveraging extends BaseSparkTest {

    public static class TestFn implements Function<LabeledPoint, LabeledPoint>{
        @Override
        public LabeledPoint call(LabeledPoint v1) throws Exception {
            return new LabeledPoint(v1.label(), Vectors.dense(v1.features().toArray()));
        }
    }

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testFromSvmLightBackprop() throws Exception {
        JavaRDD<LabeledPoint> data = MLUtils
                        .loadLibSVMFile(sc.sc(),
                                        new ClassPathResource("svmLight/iris_svmLight_0.txt").getTempFileFromArchive()
                                                        .getAbsolutePath())
                        .toJavaRDD().map(new TestFn());
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        DataSet d = new IrisDataSetIterator(150, 150).next();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(100).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.RELU).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(100).nOut(3)
                                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                                                        .build())
                        .backprop(true).build();



        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        System.out.println("Initializing network");

        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc, conf,
                        new ParameterAveragingTrainingMaster(true, numExecutors(), 1, 5, 1, 0));

        MultiLayerNetwork network2 = master.fitLabeledPoint(data);
        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());


    }


    @Test
    public void testFromSvmLight() throws Exception {
        JavaRDD<LabeledPoint> data = MLUtils
                        .loadLibSVMFile(sc.sc(),
                                        new ClassPathResource("svmLight/iris_svmLight_0.txt").getTempFileFromArchive()
                                                        .getAbsolutePath())
                        .toJavaRDD().map(new TestFn());

        DataSet d = new IrisDataSetIterator(150, 150).next();
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(123)
                                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                                        .miniBatch(true).maxNumLineSearchIterations(10)
                                        .list().layer(0,
                                                        new DenseLayer.Builder().nIn(4).nOut(100)
                                                                .weightInit(WeightInit.XAVIER)
                                                                .activation(Activation.RELU)
                                                                .build())
                                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MCXENT).nIn(100).nOut(3)
                                                                        .activation(Activation.SOFTMAX)
                                                                        .weightInit(WeightInit.XAVIER).build())
                                        .backprop(false).build();



        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc, getBasicConf(),
                        new ParameterAveragingTrainingMaster(true, numExecutors(), 1, 5, 1, 0));

        MultiLayerNetwork network2 = master.fitLabeledPoint(data);
        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());
    }

    @Test
    public void testRunIteration() {

        DataSet dataSet = new IrisDataSetIterator(5, 5).next();
        List<DataSet> list = dataSet.asList();
        JavaRDD<DataSet> data = sc.parallelize(list);

        SparkDl4jMultiLayer sparkNetCopy = new SparkDl4jMultiLayer(sc, getBasicConf(),
                        new ParameterAveragingTrainingMaster(true, numExecutors(), 1, 5, 1, 0));
        MultiLayerNetwork networkCopy = sparkNetCopy.fit(data);

        INDArray expectedParams = networkCopy.params();

        SparkDl4jMultiLayer sparkNet = getBasicNetwork();
        MultiLayerNetwork network = sparkNet.fit(data);
        INDArray actualParams = network.params();

        assertEquals(expectedParams.size(1), actualParams.size(1));
    }

    @Test
    public void testUpdaters() {
        SparkDl4jMultiLayer sparkNet = getBasicNetwork();
        MultiLayerNetwork netCopy = sparkNet.getNetwork().clone();

        netCopy.fit(data);
        IUpdater expectedUpdater = ((BaseLayer) netCopy.conf().getLayer()).getIUpdater();
        double expectedLR = ((Nesterovs)((BaseLayer) netCopy.conf().getLayer()).getIUpdater()).getLearningRate();
        double expectedMomentum = ((Nesterovs)((BaseLayer) netCopy.conf().getLayer()).getIUpdater()).getMomentum();

        IUpdater actualUpdater = ((BaseLayer) sparkNet.getNetwork().conf().getLayer()).getIUpdater();
        sparkNet.fit(sparkData);
        double actualLR = ((Nesterovs)((BaseLayer) sparkNet.getNetwork().conf().getLayer()).getIUpdater()).getLearningRate();
        double actualMomentum = ((Nesterovs)((BaseLayer) sparkNet.getNetwork().conf().getLayer()).getIUpdater()).getMomentum();

        assertEquals(expectedUpdater, actualUpdater);
        assertEquals(expectedLR, actualLR, 0.01);
        assertEquals(expectedMomentum, actualMomentum, 0.01);

    }


    @Test
    public void testEvaluation() {

        SparkDl4jMultiLayer sparkNet = getBasicNetwork();
        MultiLayerNetwork netCopy = sparkNet.getNetwork().clone();

        Evaluation evalExpected = new Evaluation();
        INDArray outLocal = netCopy.output(input, Layer.TrainingMode.TEST);
        evalExpected.eval(labels, outLocal);

        Evaluation evalActual = sparkNet.evaluate(sparkData);

        assertEquals(evalExpected.accuracy(), evalActual.accuracy(), 1e-3);
        assertEquals(evalExpected.f1(), evalActual.f1(), 1e-3);
        assertEquals(evalExpected.getNumRowCounter(), evalActual.getNumRowCounter(), 1e-3);
        assertMapEquals(evalExpected.falseNegatives(), evalActual.falseNegatives());
        assertMapEquals(evalExpected.falsePositives(), evalActual.falsePositives());
        assertMapEquals(evalExpected.trueNegatives(), evalActual.trueNegatives());
        assertMapEquals(evalExpected.truePositives(), evalActual.truePositives());
        assertEquals(evalExpected.precision(), evalActual.precision(), 1e-3);
        assertEquals(evalExpected.recall(), evalActual.recall(), 1e-3);
        assertEquals(evalExpected.getConfusionMatrix(), evalActual.getConfusionMatrix());
    }

    private static void assertMapEquals(Map<Integer, Integer> first, Map<Integer, Integer> second) {
        assertEquals(first.keySet(), second.keySet());
        for (Integer i : first.keySet()) {
            assertEquals(first.get(i), second.get(i));
        }
    }

    @Test
    public void testSmallAmountOfData() {
        //Idea: Test spark training where some executors don't get any data
        //in this case: by having fewer examples (2 DataSets) than executors (local[*])

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new RmsProp())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nIn).nOut(3)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MSE).nIn(3).nOut(nOut).activation(Activation.SOFTMAX)
                                                        .build())
                        .build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf,
                        new ParameterAveragingTrainingMaster(true, numExecutors(), 1, 10, 1, 0));

        Nd4j.getRandom().setSeed(12345);
        DataSet d1 = new DataSet(Nd4j.rand(1, nIn), Nd4j.rand(1, nOut));
        DataSet d2 = new DataSet(Nd4j.rand(1, nIn), Nd4j.rand(1, nOut));

        JavaRDD<DataSet> rddData = sc.parallelize(Arrays.asList(d1, d2));

        sparkNet.fit(rddData);

    }

    @Test
    public void testDistributedScoring() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().l1(0.1).l2(0.1)
                        .seed(123).updater(new Nesterovs(0.1, 0.9)).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nIn).nOut(3)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(3).nOut(nOut)
                                                        .activation(Activation.SOFTMAX).build())
                        .backprop(true).pretrain(false).build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf,
                        new ParameterAveragingTrainingMaster(true, numExecutors(), 1, 10, 1, 0));
        MultiLayerNetwork netCopy = sparkNet.getNetwork().clone();

        int nRows = 100;

        INDArray features = Nd4j.rand(nRows, nIn);
        INDArray labels = Nd4j.zeros(nRows, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < nRows; i++) {
            labels.putScalar(new int[] {i, r.nextInt(nOut)}, 1.0);
        }

        INDArray localScoresWithReg = netCopy.scoreExamples(new DataSet(features, labels), true);
        INDArray localScoresNoReg = netCopy.scoreExamples(new DataSet(features, labels), false);

        List<Tuple2<String, DataSet>> dataWithKeys = new ArrayList<>();
        for (int i = 0; i < nRows; i++) {
            DataSet ds = new DataSet(features.getRow(i).dup(), labels.getRow(i).dup());
            dataWithKeys.add(new Tuple2<>(String.valueOf(i), ds));
        }
        JavaPairRDD<String, DataSet> dataWithKeysRdd = sc.parallelizePairs(dataWithKeys);

        JavaPairRDD<String, Double> sparkScoresWithReg = sparkNet.scoreExamples(dataWithKeysRdd, true, 4);
        JavaPairRDD<String, Double> sparkScoresNoReg = sparkNet.scoreExamples(dataWithKeysRdd, false, 4);

        Map<String, Double> sparkScoresWithRegMap = sparkScoresWithReg.collectAsMap();
        Map<String, Double> sparkScoresNoRegMap = sparkScoresNoReg.collectAsMap();

        for (int i = 0; i < nRows; i++) {
            double scoreRegExp = localScoresWithReg.getDouble(i);
            double scoreRegAct = sparkScoresWithRegMap.get(String.valueOf(i));
            assertEquals(scoreRegExp, scoreRegAct, 1e-5);

            double scoreNoRegExp = localScoresNoReg.getDouble(i);
            double scoreNoRegAct = sparkScoresNoRegMap.get(String.valueOf(i));
            assertEquals(scoreNoRegExp, scoreNoRegAct, 1e-5);

            //            System.out.println(scoreRegExp + "\t" + scoreRegAct + "\t" + scoreNoRegExp + "\t" + scoreNoRegAct);
        }

        List<DataSet> dataNoKeys = new ArrayList<>();
        for (int i = 0; i < nRows; i++) {
            dataNoKeys.add(new DataSet(features.getRow(i).dup(), labels.getRow(i).dup()));
        }
        JavaRDD<DataSet> dataNoKeysRdd = sc.parallelize(dataNoKeys);

        List<Double> scoresWithReg = new ArrayList<>(sparkNet.scoreExamples(dataNoKeysRdd, true, 4).collect());
        List<Double> scoresNoReg = new ArrayList<>(sparkNet.scoreExamples(dataNoKeysRdd, false, 4).collect());
        Collections.sort(scoresWithReg);
        Collections.sort(scoresNoReg);
        double[] localScoresWithRegDouble = localScoresWithReg.data().asDouble();
        double[] localScoresNoRegDouble = localScoresNoReg.data().asDouble();
        Arrays.sort(localScoresWithRegDouble);
        Arrays.sort(localScoresNoRegDouble);

        for (int i = 0; i < localScoresWithRegDouble.length; i++) {
            assertEquals(localScoresWithRegDouble[i], scoresWithReg.get(i), 1e-5);
            assertEquals(localScoresNoRegDouble[i], scoresNoReg.get(i), 1e-5);

            //System.out.println(localScoresWithRegDouble[i] + "\t" + scoresWithReg.get(i) + "\t" + localScoresNoRegDouble[i] + "\t" + scoresNoReg.get(i));
        }
    }



    @Test
    public void testParameterAveragingMultipleExamplesPerDataSet() throws Exception {
        int dataSetObjSize = 5;
        int batchSizePerExecutor = 25;
        List<DataSet> list = new ArrayList<>();
        DataSetIterator iter = new MnistDataSetIterator(dataSetObjSize, 1000, false);
        while (iter.hasNext()) {
            list.add(iter.next());
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new RmsProp())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(28 * 28).nOut(50)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(50).nOut(10)
                                                        .activation(Activation.SOFTMAX).build())
                        .pretrain(false).backprop(true).build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf,
                        new ParameterAveragingTrainingMaster.Builder(numExecutors(), dataSetObjSize)
                                        .batchSizePerWorker(batchSizePerExecutor).averagingFrequency(1)
                                        .aggregationDepth(1).repartionData(Repartition.Always).build());
        sparkNet.setCollectTrainingStats(true);

        JavaRDD<DataSet> rdd = sc.parallelize(list);

        sparkNet.fit(rdd);

        SparkTrainingStats stats = sparkNet.getSparkTrainingStats();

        List<EventStats> mapPartitionStats = stats.getValue("ParameterAveragingMasterMapPartitionsTimesMs");
        int numSplits = list.size() * dataSetObjSize / (numExecutors() * batchSizePerExecutor); //For an averaging frequency of 1
        assertEquals(numSplits, mapPartitionStats.size());


        List<EventStats> workerFitStats = stats.getValue("ParameterAveragingWorkerFitTimesMs");
        for (EventStats e : workerFitStats) {
            ExampleCountEventStats eces = (ExampleCountEventStats) e;
            System.out.println(eces.getTotalExampleCount());
        }

        for (EventStats e : workerFitStats) {
            ExampleCountEventStats eces = (ExampleCountEventStats) e;
            assertEquals(batchSizePerExecutor, eces.getTotalExampleCount());
        }
    }


    @Test
    public void testFitViaStringPaths() throws Exception {

        Path tempDir = testDir.newFolder("DL4J-testFitViaStringPaths").toPath();
        File tempDirF = tempDir.toFile();
        tempDirF.deleteOnExit();

        int dataSetObjSize = 5;
        int batchSizePerExecutor = 25;
        DataSetIterator iter = new MnistDataSetIterator(dataSetObjSize, 1000, false);
        int i = 0;
        while (iter.hasNext()) {
            File nextFile = new File(tempDirF, i + ".bin");
            DataSet ds = iter.next();
            ds.save(nextFile);
            i++;
        }

        System.out.println("Saved to: " + tempDirF.getAbsolutePath());



        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new RmsProp())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(28 * 28).nOut(50)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(50).nOut(10)
                                                        .activation(Activation.SOFTMAX).build())
                        .pretrain(false).backprop(true).build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf,
                        new ParameterAveragingTrainingMaster.Builder(numExecutors(), dataSetObjSize)
                                        .workerPrefetchNumBatches(5).batchSizePerWorker(batchSizePerExecutor)
                                        .averagingFrequency(1).repartionData(Repartition.Always).build());
        sparkNet.setCollectTrainingStats(true);


        //List files:
        Configuration config = new Configuration();
        FileSystem hdfs = FileSystem.get(tempDir.toUri(), config);
        RemoteIterator<LocatedFileStatus> fileIter =
                        hdfs.listFiles(new org.apache.hadoop.fs.Path(tempDir.toString()), false);

        List<String> paths = new ArrayList<>();
        while (fileIter.hasNext()) {
            String path = fileIter.next().getPath().toString();
            paths.add(path);
        }

        INDArray paramsBefore = sparkNet.getNetwork().params().dup();
        JavaRDD<String> pathRdd = sc.parallelize(paths);
        sparkNet.fitPaths(pathRdd);

        INDArray paramsAfter = sparkNet.getNetwork().params().dup();
        assertNotEquals(paramsBefore, paramsAfter);

        SparkTrainingStats stats = sparkNet.getSparkTrainingStats();
        System.out.println(stats.statsAsString());

        sparkNet.getTrainingMaster().deleteTempFiles(sc);
    }

    @Test
    public void testFitViaStringPathsSize1() throws Exception {

        Path tempDir = testDir.newFolder("DL4J-testFitViaStringPathsSize1").toPath();
        File tempDirF = tempDir.toFile();
        tempDirF.deleteOnExit();

        int dataSetObjSize = 1;
        int batchSizePerExecutor = 25;
        int numSplits = 10;
        int averagingFrequency = 3;
        int totalExamples = numExecutors() * batchSizePerExecutor * numSplits * averagingFrequency;
        DataSetIterator iter = new MnistDataSetIterator(dataSetObjSize, totalExamples, false);
        int i = 0;
        while (iter.hasNext()) {
            File nextFile = new File(tempDirF, i + ".bin");
            DataSet ds = iter.next();
            ds.save(nextFile);
            i++;
        }

        System.out.println("Saved to: " + tempDirF.getAbsolutePath());



        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new RmsProp())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(28 * 28).nOut(50)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(50).nOut(10)
                                                        .activation(Activation.SOFTMAX).build())
                        .pretrain(false).backprop(true).build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf,
                        new ParameterAveragingTrainingMaster.Builder(numExecutors(), dataSetObjSize)
                                        .workerPrefetchNumBatches(5).batchSizePerWorker(batchSizePerExecutor)
                                        .averagingFrequency(averagingFrequency).repartionData(Repartition.Always)
                                        .build());
        sparkNet.setCollectTrainingStats(true);


        //List files:
        Configuration config = new Configuration();
        FileSystem hdfs = FileSystem.get(tempDir.toUri(), config);
        RemoteIterator<LocatedFileStatus> fileIter =
                        hdfs.listFiles(new org.apache.hadoop.fs.Path(tempDir.toString()), false);

        List<String> paths = new ArrayList<>();
        while (fileIter.hasNext()) {
            String path = fileIter.next().getPath().toString();
            paths.add(path);
        }

        INDArray paramsBefore = sparkNet.getNetwork().params().dup();
        JavaRDD<String> pathRdd = sc.parallelize(paths);
        sparkNet.fitPaths(pathRdd);

        INDArray paramsAfter = sparkNet.getNetwork().params().dup();
        assertNotEquals(paramsBefore, paramsAfter);

        Thread.sleep(2000);
        SparkTrainingStats stats = sparkNet.getSparkTrainingStats();

        //Expect
        System.out.println(stats.statsAsString());
        assertEquals(numSplits, stats.getValue("ParameterAveragingMasterRepartitionTimesMs").size());

        List<EventStats> list = stats.getValue("ParameterAveragingWorkerFitTimesMs");
        assertEquals(numSplits * numExecutors() * averagingFrequency, list.size());
        for (EventStats es : list) {
            ExampleCountEventStats e = (ExampleCountEventStats) es;
            assertTrue(batchSizePerExecutor * averagingFrequency - 10 >= e.getTotalExampleCount());
        }


        sparkNet.getTrainingMaster().deleteTempFiles(sc);
    }


    @Test
    public void testFitViaStringPathsCompGraph() throws Exception {

        Path tempDir = testDir.newFolder("DL4J-testFitViaStringPathsCG").toPath();
        Path tempDir2 = testDir.newFolder("DL4J-testFitViaStringPathsCG-MDS").toPath();
        File tempDirF = tempDir.toFile();
        File tempDirF2 = tempDir2.toFile();
        tempDirF.deleteOnExit();
        tempDirF2.deleteOnExit();

        int dataSetObjSize = 5;
        int batchSizePerExecutor = 25;
        DataSetIterator iter = new MnistDataSetIterator(dataSetObjSize, 1000, false);
        int i = 0;
        while (iter.hasNext()) {
            File nextFile = new File(tempDirF, i + ".bin");
            File nextFile2 = new File(tempDirF2, i + ".bin");
            DataSet ds = iter.next();
            MultiDataSet mds = new MultiDataSet(ds.getFeatures(), ds.getLabels());
            ds.save(nextFile);
            mds.save(nextFile2);
            i++;
        }

        System.out.println("Saved to: " + tempDirF.getAbsolutePath());
        System.out.println("Saved to: " + tempDirF2.getAbsolutePath());



        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().updater(new RmsProp())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .graphBuilder().addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(28 * 28).nOut(50)
                                        .activation(Activation.TANH).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(50).nOut(10)
                                                        .activation(Activation.SOFTMAX).build(),
                                        "0")
                        .setOutputs("1").pretrain(false).backprop(true).build();

        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, conf,
                        new ParameterAveragingTrainingMaster.Builder(numExecutors(), dataSetObjSize)
                                        .workerPrefetchNumBatches(5).workerPrefetchNumBatches(0)
                                        .batchSizePerWorker(batchSizePerExecutor).averagingFrequency(1)
                                        .repartionData(Repartition.Always).build());
        sparkNet.setCollectTrainingStats(true);


        //List files:
        Configuration config = new Configuration();
        FileSystem hdfs = FileSystem.get(tempDir.toUri(), config);
        RemoteIterator<LocatedFileStatus> fileIter =
                        hdfs.listFiles(new org.apache.hadoop.fs.Path(tempDir.toString()), false);

        List<String> paths = new ArrayList<>();
        while (fileIter.hasNext()) {
            String path = fileIter.next().getPath().toString();
            paths.add(path);
        }

        INDArray paramsBefore = sparkNet.getNetwork().params().dup();
        JavaRDD<String> pathRdd = sc.parallelize(paths);
        sparkNet.fitPaths(pathRdd);

        INDArray paramsAfter = sparkNet.getNetwork().params().dup();
        assertNotEquals(paramsBefore, paramsAfter);

        SparkTrainingStats stats = sparkNet.getSparkTrainingStats();
        System.out.println(stats.statsAsString());

        //Same thing, buf for MultiDataSet objects:
        config = new Configuration();
        hdfs = FileSystem.get(tempDir2.toUri(), config);
        fileIter = hdfs.listFiles(new org.apache.hadoop.fs.Path(tempDir2.toString()), false);

        paths = new ArrayList<>();
        while (fileIter.hasNext()) {
            String path = fileIter.next().getPath().toString();
            paths.add(path);
        }

        paramsBefore = sparkNet.getNetwork().params().dup();
        pathRdd = sc.parallelize(paths);
        sparkNet.fitPathsMultiDataSet(pathRdd);

        paramsAfter = sparkNet.getNetwork().params().dup();
        assertNotEquals(paramsBefore, paramsAfter);

        stats = sparkNet.getSparkTrainingStats();
        System.out.println(stats.statsAsString());
    }


    @Test
    public void testSeedRepeatability() throws Exception {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(new RmsProp())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(4).nOut(4)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(4).nOut(3).activation(Activation.SOFTMAX)
                                                        .build())
                        .pretrain(false).backprop(true).build();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork n1 = new MultiLayerNetwork(conf);
        n1.init();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork n2 = new MultiLayerNetwork(conf);
        n2.init();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork n3 = new MultiLayerNetwork(conf);
        n3.init();

        SparkDl4jMultiLayer sparkNet1 = new SparkDl4jMultiLayer(sc, n1,
                        new ParameterAveragingTrainingMaster.Builder(1).workerPrefetchNumBatches(5)
                                        .batchSizePerWorker(5).averagingFrequency(1).repartionData(Repartition.Always)
                                        .rngSeed(12345).build());

        Thread.sleep(100); //Training master IDs are only unique if they are created at least 1 ms apart...

        SparkDl4jMultiLayer sparkNet2 = new SparkDl4jMultiLayer(sc, n2,
                        new ParameterAveragingTrainingMaster.Builder(1).workerPrefetchNumBatches(5)
                                        .batchSizePerWorker(5).averagingFrequency(1).repartionData(Repartition.Always)
                                        .rngSeed(12345).build());

        Thread.sleep(100);

        SparkDl4jMultiLayer sparkNet3 = new SparkDl4jMultiLayer(sc, n3,
                        new ParameterAveragingTrainingMaster.Builder(1).workerPrefetchNumBatches(5)
                                        .batchSizePerWorker(5).averagingFrequency(1).repartionData(Repartition.Always)
                                        .rngSeed(98765).build());

        List<DataSet> data = new ArrayList<>();
        DataSetIterator iter = new IrisDataSetIterator(1, 150);
        while (iter.hasNext())
            data.add(iter.next());

        JavaRDD<DataSet> rdd = sc.parallelize(data);


        sparkNet1.fit(rdd);
        sparkNet2.fit(rdd);
        sparkNet3.fit(rdd);


        INDArray p1 = sparkNet1.getNetwork().params();
        INDArray p2 = sparkNet2.getNetwork().params();
        INDArray p3 = sparkNet3.getNetwork().params();

        sparkNet1.getTrainingMaster().deleteTempFiles(sc);
        sparkNet2.getTrainingMaster().deleteTempFiles(sc);
        sparkNet3.getTrainingMaster().deleteTempFiles(sc);

        assertEquals(p1, p2);
        assertNotEquals(p1, p3);
    }


    @Test
    public void testIterationCounts() throws Exception {
        int dataSetObjSize = 5;
        int batchSizePerExecutor = 25;
        List<DataSet> list = new ArrayList<>();
        int minibatchesPerWorkerPerEpoch = 10;
        DataSetIterator iter = new MnistDataSetIterator(dataSetObjSize,
                        batchSizePerExecutor * numExecutors() * minibatchesPerWorkerPerEpoch, false);
        while (iter.hasNext()) {
            list.add(iter.next());
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new RmsProp())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(28 * 28).nOut(50)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(50).nOut(10)
                                                        .activation(Activation.SOFTMAX).build())
                        .pretrain(false).backprop(true).build();

        for (int avgFreq : new int[] {1, 5, 10}) {
            System.out.println("--- Avg freq " + avgFreq + " ---");
            SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf.clone(),
                            new ParameterAveragingTrainingMaster.Builder(numExecutors(), dataSetObjSize)
                                            .batchSizePerWorker(batchSizePerExecutor).averagingFrequency(avgFreq)
                                            .repartionData(Repartition.Always).build());

            sparkNet.setListeners(new ScoreIterationListener(1));



            JavaRDD<DataSet> rdd = sc.parallelize(list);

            assertEquals(0, sparkNet.getNetwork().getLayerWiseConfigurations().getIterationCount());
            sparkNet.fit(rdd);
            assertEquals(minibatchesPerWorkerPerEpoch,
                            sparkNet.getNetwork().getLayerWiseConfigurations().getIterationCount());
            sparkNet.fit(rdd);
            assertEquals(2 * minibatchesPerWorkerPerEpoch,
                            sparkNet.getNetwork().getLayerWiseConfigurations().getIterationCount());

            sparkNet.getTrainingMaster().deleteTempFiles(sc);
        }
    }

    @Test
    public void testIterationCountsGraph() throws Exception {
        int dataSetObjSize = 5;
        int batchSizePerExecutor = 25;
        List<DataSet> list = new ArrayList<>();
        int minibatchesPerWorkerPerEpoch = 10;
        DataSetIterator iter = new MnistDataSetIterator(dataSetObjSize,
                        batchSizePerExecutor * numExecutors() * minibatchesPerWorkerPerEpoch, false);
        while (iter.hasNext()) {
            list.add(iter.next());
        }

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().updater(new RmsProp())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .graphBuilder().addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(28 * 28).nOut(50)
                                        .activation(Activation.TANH).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(50).nOut(10)
                                                        .activation(Activation.SOFTMAX).build(),
                                        "0")
                        .pretrain(false).backprop(true).setOutputs("1").build();

        for (int avgFreq : new int[] {1, 5, 10}) {
            System.out.println("--- Avg freq " + avgFreq + " ---");
            SparkComputationGraph sparkNet = new SparkComputationGraph(sc, conf.clone(),
                            new ParameterAveragingTrainingMaster.Builder(numExecutors(), dataSetObjSize)
                                            .batchSizePerWorker(batchSizePerExecutor).averagingFrequency(avgFreq)
                                            .repartionData(Repartition.Always).build());

            sparkNet.setListeners(new ScoreIterationListener(1));

            JavaRDD<DataSet> rdd = sc.parallelize(list);

            assertEquals(0, sparkNet.getNetwork().getConfiguration().getIterationCount());
            sparkNet.fit(rdd);
            assertEquals(minibatchesPerWorkerPerEpoch, sparkNet.getNetwork().getConfiguration().getIterationCount());
            sparkNet.fit(rdd);
            assertEquals(2 * minibatchesPerWorkerPerEpoch,
                            sparkNet.getNetwork().getConfiguration().getIterationCount());

            sparkNet.getTrainingMaster().deleteTempFiles(sc);
        }
    }


    @Test
    public void testVaePretrainSimple() {
        //Simple sanity check on pretraining
        int nIn = 8;

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(new RmsProp())
                        .weightInit(WeightInit.XAVIER).list()
                        .layer(0, new VariationalAutoencoder.Builder().nIn(8).nOut(10).encoderLayerSizes(12)
                                        .decoderLayerSizes(13).reconstructionDistribution(
                                                        new GaussianReconstructionDistribution(Activation.IDENTITY))
                                        .build())
                        .pretrain(true).backprop(false).build();

        //Do training on Spark with one executor, for 3 separate minibatches
        int rddDataSetNumExamples = 10;
        int totalAveragings = 5;
        int averagingFrequency = 3;
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(rddDataSetNumExamples)
                        .averagingFrequency(averagingFrequency).batchSizePerWorker(rddDataSetNumExamples)
                        .saveUpdater(true).workerPrefetchNumBatches(0).build();
        Nd4j.getRandom().setSeed(12345);
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf.clone(), tm);

        List<DataSet> trainData = new ArrayList<>();
        int nDataSets = numExecutors() * totalAveragings * averagingFrequency;
        for (int i = 0; i < nDataSets; i++) {
            trainData.add(new DataSet(Nd4j.rand(rddDataSetNumExamples, nIn), null));
        }

        JavaRDD<DataSet> data = sc.parallelize(trainData);

        sparkNet.fit(data);
    }

    @Test
    public void testVaePretrainSimpleCG() {
        //Simple sanity check on pretraining
        int nIn = 8;

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(new RmsProp())
                        .weightInit(WeightInit.XAVIER).graphBuilder().addInputs("in")
                        .addLayer("0", new VariationalAutoencoder.Builder().nIn(8).nOut(10).encoderLayerSizes(12)
                                        .decoderLayerSizes(13).reconstructionDistribution(
                                                        new GaussianReconstructionDistribution(Activation.IDENTITY))
                                        .build(), "in")
                        .setOutputs("0").pretrain(true).backprop(false).build();

        //Do training on Spark with one executor, for 3 separate minibatches
        int rddDataSetNumExamples = 10;
        int totalAveragings = 5;
        int averagingFrequency = 3;
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(rddDataSetNumExamples)
                        .averagingFrequency(averagingFrequency).batchSizePerWorker(rddDataSetNumExamples)
                        .saveUpdater(true).workerPrefetchNumBatches(0).build();
        Nd4j.getRandom().setSeed(12345);
        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, conf.clone(), tm);

        List<DataSet> trainData = new ArrayList<>();
        int nDataSets = numExecutors() * totalAveragings * averagingFrequency;
        for (int i = 0; i < nDataSets; i++) {
            trainData.add(new DataSet(Nd4j.rand(rddDataSetNumExamples, nIn), null));
        }

        JavaRDD<DataSet> data = sc.parallelize(trainData);

        sparkNet.fit(data);
    }


    @Test
    public void testROC() {

        int nArrays = 100;
        int minibatch = 64;
        int steps = 20;
        int nIn = 5;
        int nOut = 2;
        int layerSize = 10;

        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).list()
                                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(layerSize).build())
                                        .layer(1, new OutputLayer.Builder().nIn(layerSize).nOut(nOut)
                                                        .activation(Activation.SOFTMAX).lossFunction(
                                                                        LossFunctions.LossFunction.MCXENT)
                                                        .build())
                                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);

        ROC local = new ROC(steps);
        List<DataSet> dsList = new ArrayList<>();
        for (int i = 0; i < nArrays; i++) {
            INDArray features = Nd4j.rand(minibatch, nIn);

            INDArray p = net.output(features);

            INDArray l = Nd4j.zeros(minibatch, 2);
            for (int j = 0; j < minibatch; j++) {
                l.putScalar(j, r.nextInt(2), 1.0);
            }

            local.eval(l, p);

            dsList.add(new DataSet(features, l));
        }


        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, net, null);
        JavaRDD<DataSet> rdd = sc.parallelize(dsList);

        ROC sparkROC = sparkNet.evaluateROC(rdd, steps, 32);

        assertEquals(sparkROC.calculateAUC(), sparkROC.calculateAUC(), 1e-6);

        assertEquals(local.getRocCurve(), sparkROC.getRocCurve());
    }


    @Test
    public void testROCMultiClass() {

        int nArrays = 100;
        int minibatch = 64;
        int steps = 20;
        int nIn = 5;
        int nOut = 3;
        int layerSize = 10;

        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).list()
                                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(layerSize).build())
                                        .layer(1, new OutputLayer.Builder().nIn(layerSize).nOut(nOut)
                                                        .activation(Activation.SOFTMAX).lossFunction(
                                                                        LossFunctions.LossFunction.MCXENT)
                                                        .build())
                                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);

        ROCMultiClass local = new ROCMultiClass(steps);
        List<DataSet> dsList = new ArrayList<>();
        for (int i = 0; i < nArrays; i++) {
            INDArray features = Nd4j.rand(minibatch, nIn);

            INDArray p = net.output(features);

            INDArray l = Nd4j.zeros(minibatch, nOut);
            for (int j = 0; j < minibatch; j++) {
                l.putScalar(j, r.nextInt(nOut), 1.0);
            }

            local.eval(l, p);

            dsList.add(new DataSet(features, l));
        }


        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, net, null);
        JavaRDD<DataSet> rdd = sc.parallelize(dsList);

        ROCMultiClass sparkROC = sparkNet.evaluateROCMultiClass(rdd, steps, 32);

        for (int i = 0; i < nOut; i++) {
            assertEquals(sparkROC.calculateAUC(i), sparkROC.calculateAUC(i), 1e-6);

            assertEquals(local.getRocCurve(i), sparkROC.getRocCurve(i));
        }
    }


    @Test
    public void testEpochCounter() throws Exception {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new OutputLayer.Builder().nIn(4).nOut(3).build())
                .build();

        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("out", new OutputLayer.Builder().nIn(4).nOut(3).build(), "in")
                .setOutputs("out")
                .build();

        DataSetIterator iter = new IrisDataSetIterator(1, 150);

        List<DataSet> l = new ArrayList<>();
        while(iter.hasNext()){
            l.add(iter.next());
        }

        JavaRDD<DataSet> rdd = sc.parallelize(l);


        int rddDataSetNumExamples = 1;
        int averagingFrequency = 3;
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(rddDataSetNumExamples)
                .averagingFrequency(averagingFrequency).batchSizePerWorker(rddDataSetNumExamples)
                .saveUpdater(true).workerPrefetchNumBatches(0).build();
        Nd4j.getRandom().setSeed(12345);


        SparkDl4jMultiLayer sn1 = new SparkDl4jMultiLayer(sc, conf.clone(), tm);
        SparkComputationGraph sn2 = new SparkComputationGraph(sc, conf2.clone(), tm);


        for(int i=0; i<4; i++ ){
            assertEquals(i, sn1.getNetwork().getLayerWiseConfigurations().getEpochCount());
            assertEquals(i, sn2.getNetwork().getConfiguration().getEpochCount());
            sn1.fit(rdd);
            sn2.fit(rdd);
            assertEquals(i+1, sn1.getNetwork().getLayerWiseConfigurations().getEpochCount());
            assertEquals(i+1, sn2.getNetwork().getConfiguration().getEpochCount());
        }
    }
}

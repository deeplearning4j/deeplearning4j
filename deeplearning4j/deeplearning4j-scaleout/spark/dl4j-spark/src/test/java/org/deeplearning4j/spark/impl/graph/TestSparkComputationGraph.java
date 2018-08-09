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

package org.deeplearning4j.spark.impl.graph;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class TestSparkComputationGraph extends BaseSparkTest {

    public static ComputationGraph getBasicNetIris2Class() {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                        .graphBuilder().addInputs("in")
                        .addLayer("l0", new DenseLayer.Builder().nIn(4).nOut(10).build(), "in")
                        .addLayer("l1", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(10).nOut(2).build(), "l0")
                        .setOutputs("l1").build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        return cg;
    }

    @Test
    public void testBasic() throws Exception {

        JavaSparkContext sc = this.sc;

        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));
        MultiDataSetIterator iter = new RecordReaderMultiDataSetIterator.Builder(1).addReader("iris", rr)
                        .addInput("iris", 0, 3).addOutputOneHot("iris", 4, 3).build();

        List<MultiDataSet> list = new ArrayList<>(150);
        while (iter.hasNext())
            list.add(iter.next());

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                        .updater(new Sgd(0.1))
                        .graphBuilder().addInputs("in")
                        .addLayer("dense", new DenseLayer.Builder().nIn(4).nOut(2).build(), "in").addLayer("out",
                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3)
                                                        .build(),
                                        "dense")
                        .setOutputs("out").pretrain(false).backprop(true).build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        TrainingMaster tm = new ParameterAveragingTrainingMaster(true, numExecutors(), 1, 10, 1, 0);

        SparkComputationGraph scg = new SparkComputationGraph(sc, cg, tm);
        scg.setListeners(Collections.singleton((TrainingListener) new ScoreIterationListener(1)));

        JavaRDD<MultiDataSet> rdd = sc.parallelize(list);
        scg.fitMultiDataSet(rdd);

        //Try: fitting using DataSet
        DataSetIterator iris = new IrisDataSetIterator(1, 150);
        List<DataSet> list2 = new ArrayList<>();
        while (iris.hasNext())
            list2.add(iris.next());
        JavaRDD<DataSet> rddDS = sc.parallelize(list2);

        scg.fit(rddDS);
    }


    @Test
    public void testDistributedScoring() {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().l1(0.1).l2(0.1)
                        .seed(123).updater(new Nesterovs(0.1, 0.9)).graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nIn).nOut(3)
                                        .activation(Activation.TANH).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(3).nOut(nOut)
                                                        .activation(Activation.SOFTMAX).build(),
                                        "0")
                        .setOutputs("1").backprop(true).pretrain(false).build();

        TrainingMaster tm = new ParameterAveragingTrainingMaster(true, numExecutors(), 1, 10, 1, 0);

        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, conf, tm);
        ComputationGraph netCopy = sparkNet.getNetwork().clone();

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

            //            System.out.println(localScoresWithRegDouble[i] + "\t" + scoresWithReg.get(i) + "\t" + localScoresNoRegDouble[i] + "\t" + scoresNoReg.get(i));
        }
    }

    @Test
    public void testSeedRepeatability() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(Updater.RMSPROP)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER).graphBuilder().addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(4).nOut(4)
                                        .activation(Activation.TANH).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(4).nOut(3).activation(Activation.SOFTMAX)
                                                        .build(),
                                        "0")
                        .setOutputs("1").pretrain(false).backprop(true).build();

        Nd4j.getRandom().setSeed(12345);
        ComputationGraph n1 = new ComputationGraph(conf.clone());
        n1.init();

        Nd4j.getRandom().setSeed(12345);
        ComputationGraph n2 = new ComputationGraph(conf.clone());
        n2.init();

        Nd4j.getRandom().setSeed(12345);
        ComputationGraph n3 = new ComputationGraph(conf.clone());
        n3.init();

        SparkComputationGraph sparkNet1 = new SparkComputationGraph(sc, n1,
                        new ParameterAveragingTrainingMaster.Builder(1).workerPrefetchNumBatches(5)
                                        .batchSizePerWorker(5).averagingFrequency(1).repartionData(Repartition.Always)
                                        .rngSeed(12345).build());

        Thread.sleep(100); //Training master IDs are only unique if they are created at least 1 ms apart...

        SparkComputationGraph sparkNet2 = new SparkComputationGraph(sc, n2,
                        new ParameterAveragingTrainingMaster.Builder(1).workerPrefetchNumBatches(5)
                                        .batchSizePerWorker(5).averagingFrequency(1).repartionData(Repartition.Always)
                                        .rngSeed(12345).build());

        Thread.sleep(100);

        SparkComputationGraph sparkNet3 = new SparkComputationGraph(sc, n3,
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


    @Test(timeout = 60000L)
    public void testEvaluationAndRoc() {
        for( int evalWorkers : new int[]{1, 4, 8}) {
            DataSetIterator iter = new IrisDataSetIterator(5, 150);

            //Make a 2-class version of iris:
            List<DataSet> l = new ArrayList<>();
            iter.reset();
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                INDArray newL = Nd4j.create(ds.getLabels().size(0), 2);
                newL.putColumn(0, ds.getLabels().getColumn(0));
                newL.putColumn(1, ds.getLabels().getColumn(1));
                newL.getColumn(1).addi(ds.getLabels().getColumn(2));
                ds.setLabels(newL);
                l.add(ds);
            }

            iter = new ListDataSetIterator<>(l);

            ComputationGraph cg = getBasicNetIris2Class();

            Evaluation e = cg.evaluate(iter);
            ROC roc = cg.evaluateROC(iter, 32);


            SparkComputationGraph scg = new SparkComputationGraph(sc, cg, null);
            scg.setDefaultEvaluationWorkers(evalWorkers);


            JavaRDD<DataSet> rdd = sc.parallelize(l);
            rdd = rdd.repartition(20);

            Evaluation e2 = scg.evaluate(rdd);
            ROC roc2 = scg.evaluateROC(rdd);


            assertEquals(e2.accuracy(), e.accuracy(), 1e-3);
            assertEquals(e2.f1(), e.f1(), 1e-3);
            assertEquals(e2.getNumRowCounter(), e.getNumRowCounter(), 1e-3);
            assertEquals(e2.falseNegatives(), e.falseNegatives());
            assertEquals(e2.falsePositives(), e.falsePositives());
            assertEquals(e2.trueNegatives(), e.trueNegatives());
            assertEquals(e2.truePositives(), e.truePositives());
            assertEquals(e2.precision(), e.precision(), 1e-3);
            assertEquals(e2.recall(), e.recall(), 1e-3);
            assertEquals(e2.getConfusionMatrix(), e.getConfusionMatrix());

            assertEquals(roc.calculateAUC(), roc2.calculateAUC(), 1e-5);
            assertEquals(roc.calculateAUCPR(), roc2.calculateAUCPR(), 1e-5);
        }
    }

    @Test
    public void testEvaluationAndRocMDS() {
        for( int evalWorkers : new int[]{1, 4, 8}) {

            DataSetIterator iter = new IrisDataSetIterator(5, 150);

            //Make a 2-class version of iris:
            List<MultiDataSet> l = new ArrayList<>();
            iter.reset();
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                INDArray newL = Nd4j.create(ds.getLabels().size(0), 2);
                newL.putColumn(0, ds.getLabels().getColumn(0));
                newL.putColumn(1, ds.getLabels().getColumn(1));
                newL.getColumn(1).addi(ds.getLabels().getColumn(2));

                MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), newL);
                l.add(mds);
            }

            MultiDataSetIterator mdsIter = new IteratorMultiDataSetIterator(l.iterator(), 5);

            ComputationGraph cg = getBasicNetIris2Class();

            IEvaluation[] es = cg.doEvaluation(mdsIter, new Evaluation(), new ROC(32));
            Evaluation e = (Evaluation) es[0];
            ROC roc = (ROC) es[1];


            SparkComputationGraph scg = new SparkComputationGraph(sc, cg, null);
            scg.setDefaultEvaluationWorkers(evalWorkers);

            JavaRDD<MultiDataSet> rdd = sc.parallelize(l);
            rdd = rdd.repartition(20);

            IEvaluation[] es2 = scg.doEvaluationMDS(rdd, 5, new Evaluation(), new ROC(32));
            Evaluation e2 = (Evaluation) es2[0];
            ROC roc2 = (ROC) es2[1];


            assertEquals(e2.accuracy(), e.accuracy(), 1e-3);
            assertEquals(e2.f1(), e.f1(), 1e-3);
            assertEquals(e2.getNumRowCounter(), e.getNumRowCounter(), 1e-3);
            assertEquals(e2.falseNegatives(), e.falseNegatives());
            assertEquals(e2.falsePositives(), e.falsePositives());
            assertEquals(e2.trueNegatives(), e.trueNegatives());
            assertEquals(e2.truePositives(), e.truePositives());
            assertEquals(e2.precision(), e.precision(), 1e-3);
            assertEquals(e2.recall(), e.recall(), 1e-3);
            assertEquals(e2.getConfusionMatrix(), e.getConfusionMatrix());

            assertEquals(roc.calculateAUC(), roc2.calculateAUC(), 1e-5);
            assertEquals(roc.calculateAUCPR(), roc2.calculateAUCPR(), 1e-5);
        }
    }

}

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

package org.deeplearning4j.eval;

import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.SingletonMultiDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.common.resources.Resources;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 12/22/14.
 */
public class EvalTest extends BaseDL4JTest {

    @Test
    public void testIris() {

        // Network config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).seed(42)
                        .updater(new Sgd(1e-6)).list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(2).activation(Activation.TANH)
                                        .weightInit(WeightInit.XAVIER).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.SOFTMAX).build())

                        .build();

        // Instantiate model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.addListeners(new ScoreIterationListener(1));

        // Train-test split
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        DataSet next = iter.next();
        next.shuffle();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(5, new Random(42));

        // Train
        DataSet train = trainTest.getTrain();
        train.normalizeZeroMeanZeroUnitVariance();

        // Test
        DataSet test = trainTest.getTest();
        test.normalizeZeroMeanZeroUnitVariance();
        INDArray testFeature = test.getFeatures();
        INDArray testLabel = test.getLabels();

        // Fitting model
        model.fit(train);
        // Get predictions from test feature
        INDArray testPredictedLabel = model.output(testFeature);

        // Eval with class number
        org.nd4j.evaluation.classification.Evaluation eval = new org.nd4j.evaluation.classification.Evaluation(3); //// Specify class num here
        eval.eval(testLabel, testPredictedLabel);
        double eval1F1 = eval.f1();
        double eval1Acc = eval.accuracy();

        // Eval without class number
        org.nd4j.evaluation.classification.Evaluation eval2 = new org.nd4j.evaluation.classification.Evaluation(); //// No class num
        eval2.eval(testLabel, testPredictedLabel);
        double eval2F1 = eval2.f1();
        double eval2Acc = eval2.accuracy();

        //Assert the two implementations give same f1 and accuracy (since one batch)
        assertTrue(eval1F1 == eval2F1 && eval1Acc == eval2Acc);

        org.nd4j.evaluation.classification.Evaluation evalViaMethod = model.evaluate(new ListDataSetIterator<>(Collections.singletonList(test)));
        checkEvaluationEquality(eval, evalViaMethod);

//        System.out.println(eval.getConfusionMatrix().toString());
//        System.out.println(eval.getConfusionMatrix().toCSV());
//        System.out.println(eval.getConfusionMatrix().toHTML());
//        System.out.println(eval.confusionToString());

        eval.getConfusionMatrix().toString();
        eval.getConfusionMatrix().toCSV();
        eval.getConfusionMatrix().toHTML();
        eval.confusionToString();
    }

    private static void assertMapEquals(Map<Integer, Integer> first, Map<Integer, Integer> second) {
        assertEquals(first.keySet(), second.keySet());
        for (Integer i : first.keySet()) {
            assertEquals(first.get(i), second.get(i));
        }
    }

    private static void checkEvaluationEquality(org.nd4j.evaluation.classification.Evaluation evalExpected, org.nd4j.evaluation.classification.Evaluation evalActual) {
        assertEquals(evalExpected.accuracy(), evalActual.accuracy(), 1e-3);
        assertEquals(evalExpected.f1(), evalActual.f1(), 1e-3);
        assertEquals(evalExpected.getNumRowCounter(), evalActual.getNumRowCounter(), 1e-3);
        assertMapEquals(evalExpected.falseNegatives(), evalActual.falseNegatives());
        assertMapEquals(evalExpected.falsePositives(), evalActual.falsePositives());
        assertMapEquals(evalExpected.trueNegatives(), evalActual.trueNegatives());
        assertMapEquals(evalExpected.truePositives(), evalActual.truePositives());
        assertEquals(evalExpected.precision(), evalActual.precision(), 1e-3);
        assertEquals(evalExpected.recall(), evalActual.recall(), 1e-3);
        assertEquals(evalExpected.falsePositiveRate(), evalActual.falsePositiveRate(), 1e-3);
        assertEquals(evalExpected.falseNegativeRate(), evalActual.falseNegativeRate(), 1e-3);
        assertEquals(evalExpected.falseAlarmRate(), evalActual.falseAlarmRate(), 1e-3);
        assertEquals(evalExpected.getConfusionMatrix(), evalActual.getConfusionMatrix());
    }

    @Test
    public void testEvaluationWithMetaData() throws Exception {

        RecordReader csv = new CSVRecordReader();
        csv.initialize(new FileSplit(Resources.asFile("iris.txt")));

        int batchSize = 10;
        int labelIdx = 4;
        int numClasses = 3;

        RecordReaderDataSetIterator rrdsi = new RecordReaderDataSetIterator(csv, batchSize, labelIdx, numClasses);

        NormalizerStandardize ns = new NormalizerStandardize();
        ns.fit(rrdsi);
        rrdsi.setPreProcessor(ns);
        rrdsi.reset();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Sgd(0.1))
                        .list()
                        .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(4).nOut(3).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        for (int i = 0; i < 4; i++) {
            net.fit(rrdsi);
            rrdsi.reset();
        }

        org.nd4j.evaluation.classification.Evaluation e = new org.nd4j.evaluation.classification.Evaluation();
        rrdsi.setCollectMetaData(true); //*** New: Enable collection of metadata (stored in the DataSets) ***

        while (rrdsi.hasNext()) {
            DataSet ds = rrdsi.next();
            List<RecordMetaData> meta = ds.getExampleMetaData(RecordMetaData.class); //*** New - cross dependencies here make types difficult, usid Object internally in DataSet for this***

            INDArray out = net.output(ds.getFeatures());
            e.eval(ds.getLabels(), out, meta); //*** New - evaluate and also store metadata ***
        }

//        System.out.println(e.stats());
        e.stats();

//        System.out.println("\n\n*** Prediction Errors: ***");

        List<org.nd4j.evaluation.meta.Prediction> errors = e.getPredictionErrors(); //*** New - get list of prediction errors from evaluation ***
        List<RecordMetaData> metaForErrors = new ArrayList<>();
        for (org.nd4j.evaluation.meta.Prediction p : errors) {
            metaForErrors.add((RecordMetaData) p.getRecordMetaData());
        }
        DataSet ds = rrdsi.loadFromMetaData(metaForErrors); //*** New - dynamically load a subset of the data, just for prediction errors ***
        INDArray output = net.output(ds.getFeatures());

        int count = 0;
        for (org.nd4j.evaluation.meta.Prediction t : errors) {
            String s = t + "\t\tRaw Data: "
                    + csv.loadFromMetaData((RecordMetaData) t.getRecordMetaData()).getRecord() //*** New - load subset of data from MetaData object (usually batched for efficiency) ***
                    + "\tNormalized: " + ds.getFeatures().getRow(count) + "\tLabels: "
                    + ds.getLabels().getRow(count) + "\tNetwork predictions: " + output.getRow(count);
//            System.out.println(s);
            count++;
        }

        int errorCount = errors.size();
        double expAcc = 1.0 - errorCount / 150.0;
        assertEquals(expAcc, e.accuracy(), 1e-5);

        org.nd4j.evaluation.classification.ConfusionMatrix<Integer> confusion = e.getConfusionMatrix();
        int[] actualCounts = new int[3];
        int[] predictedCounts = new int[3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int entry = confusion.getCount(i, j); //(actual,predicted)
                List<org.nd4j.evaluation.meta.Prediction> list = e.getPredictions(i, j);
                assertEquals(entry, list.size());

                actualCounts[i] += entry;
                predictedCounts[j] += entry;
            }
        }

        for (int i = 0; i < 3; i++) {
            List<org.nd4j.evaluation.meta.Prediction> actualClassI = e.getPredictionsByActualClass(i);
            List<org.nd4j.evaluation.meta.Prediction> predictedClassI = e.getPredictionByPredictedClass(i);
            assertEquals(actualCounts[i], actualClassI.size());
            assertEquals(predictedCounts[i], predictedClassI.size());
        }


        //Finally: test doEvaluation methods
        rrdsi.reset();
        org.nd4j.evaluation.classification.Evaluation e2 = new org.nd4j.evaluation.classification.Evaluation();
        net.doEvaluation(rrdsi, e2);
        for (int i = 0; i < 3; i++) {
            List<org.nd4j.evaluation.meta.Prediction> actualClassI = e2.getPredictionsByActualClass(i);
            List<org.nd4j.evaluation.meta.Prediction> predictedClassI = e2.getPredictionByPredictedClass(i);
            assertEquals(actualCounts[i], actualClassI.size());
            assertEquals(predictedCounts[i], predictedClassI.size());
        }

        ComputationGraph cg = net.toComputationGraph();
        rrdsi.reset();
        e2 = new org.nd4j.evaluation.classification.Evaluation();
        cg.doEvaluation(rrdsi, e2);
        for (int i = 0; i < 3; i++) {
            List<org.nd4j.evaluation.meta.Prediction> actualClassI = e2.getPredictionsByActualClass(i);
            List<org.nd4j.evaluation.meta.Prediction> predictedClassI = e2.getPredictionByPredictedClass(i);
            assertEquals(actualCounts[i], actualClassI.size());
            assertEquals(predictedCounts[i], predictedClassI.size());
        }

    }

    private static void apply(org.nd4j.evaluation.classification.Evaluation e, int nTimes, INDArray predicted, INDArray actual) {
        for (int i = 0; i < nTimes; i++) {
            e.eval(actual, predicted);
        }
    }

    @Test
    public void testEvalSplitting(){
        //Test for "tbptt-like" functionality

        for(WorkspaceMode ws : WorkspaceMode.values()) {
            System.out.println("Starting test for workspace mode: " + ws);

            int nIn = 4;
            int layerSize = 5;
            int nOut = 6;
            int tbpttLength = 10;
            int tsLength = 5 * tbpttLength + tbpttLength / 2;

            MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .trainingWorkspaceMode(ws)
                    .inferenceWorkspaceMode(ws)
                    .list()
                    .layer(new LSTM.Builder().nIn(nIn).nOut(layerSize).build())
                    .layer(new RnnOutputLayer.Builder().nIn(layerSize).nOut(nOut)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .build();

            MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .trainingWorkspaceMode(ws)
                    .inferenceWorkspaceMode(ws)
                    .list()
                    .layer(new LSTM.Builder().nIn(nIn).nOut(layerSize).build())
                    .layer(new RnnOutputLayer.Builder().nIn(layerSize).nOut(nOut)
                            .activation(Activation.SOFTMAX).build())
                    .tBPTTLength(10)
                    .backpropType(BackpropType.TruncatedBPTT)
                    .build();

            MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
            net1.init();

            MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
            net2.init();

            net2.setParams(net1.params());

            for(boolean useMask : new boolean[]{false, true}) {

                INDArray in1 = Nd4j.rand(new int[]{3, nIn, tsLength});
                INDArray out1 = TestUtils.randomOneHotTimeSeries(3, nOut, tsLength);

                INDArray in2 = Nd4j.rand(new int[]{5, nIn, tsLength});
                INDArray out2 = TestUtils.randomOneHotTimeSeries(5, nOut, tsLength);

                INDArray lMask1 = null;
                INDArray lMask2 = null;
                if(useMask){
                    lMask1 = Nd4j.create(3, tsLength);
                    lMask2 = Nd4j.create(5, tsLength);
                    Nd4j.getExecutioner().exec(new BernoulliDistribution(lMask1, 0.5));
                    Nd4j.getExecutioner().exec(new BernoulliDistribution(lMask2, 0.5));
                }

                List<DataSet> l = Arrays.asList(new DataSet(in1, out1, null, lMask1), new DataSet(in2, out2, null, lMask2));
                DataSetIterator iter = new ExistingDataSetIterator(l);

//                System.out.println("Net 1 eval");
                org.nd4j.evaluation.IEvaluation[] e1 = net1.doEvaluation(iter, new org.nd4j.evaluation.classification.Evaluation(), new org.nd4j.evaluation.classification.ROCMultiClass(), new org.nd4j.evaluation.regression.RegressionEvaluation());
//                System.out.println("Net 2 eval");
                org.nd4j.evaluation.IEvaluation[] e2 = net2.doEvaluation(iter, new org.nd4j.evaluation.classification.Evaluation(), new org.nd4j.evaluation.classification.ROCMultiClass(), new org.nd4j.evaluation.regression.RegressionEvaluation());

                assertEquals(e1[0], e2[0]);
                assertEquals(e1[1], e2[1]);
                assertEquals(e1[2], e2[2]);
            }
        }
    }

    @Test
    public void testEvalSplittingCompGraph(){
        //Test for "tbptt-like" functionality

        for(WorkspaceMode ws : WorkspaceMode.values()) {
            System.out.println("Starting test for workspace mode: " + ws);

            int nIn = 4;
            int layerSize = 5;
            int nOut = 6;
            int tbpttLength = 10;
            int tsLength = 5 * tbpttLength + tbpttLength / 2;

            ComputationGraphConfiguration conf1 = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .trainingWorkspaceMode(ws)
                    .inferenceWorkspaceMode(ws)
                    .graphBuilder()
                    .addInputs("in")
                    .addLayer("0", new LSTM.Builder().nIn(nIn).nOut(layerSize).build(), "in")
                    .addLayer("1", new RnnOutputLayer.Builder().nIn(layerSize).nOut(nOut)
                            .activation(Activation.SOFTMAX)
                            .build(), "0")
                    .setOutputs("1")
                    .build();

            ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .trainingWorkspaceMode(ws)
                    .inferenceWorkspaceMode(ws)
                    .graphBuilder()
                    .addInputs("in")
                    .addLayer("0", new LSTM.Builder().nIn(nIn).nOut(layerSize).build(), "in")
                    .addLayer("1", new RnnOutputLayer.Builder().nIn(layerSize).nOut(nOut)
                            .activation(Activation.SOFTMAX)
                            .build(), "0")
                    .setOutputs("1")
                    .tBPTTLength(10)
                    .backpropType(BackpropType.TruncatedBPTT)
                    .build();

            ComputationGraph net1 = new ComputationGraph(conf1);
            net1.init();

            ComputationGraph net2 = new ComputationGraph(conf2);
            net2.init();

            net2.setParams(net1.params());

            for (boolean useMask : new boolean[]{false, true}) {

                INDArray in1 = Nd4j.rand(new int[]{3, nIn, tsLength});
                INDArray out1 = TestUtils.randomOneHotTimeSeries(3, nOut, tsLength);

                INDArray in2 = Nd4j.rand(new int[]{5, nIn, tsLength});
                INDArray out2 = TestUtils.randomOneHotTimeSeries(5, nOut, tsLength);

                INDArray lMask1 = null;
                INDArray lMask2 = null;
                if (useMask) {
                    lMask1 = Nd4j.create(3, tsLength);
                    lMask2 = Nd4j.create(5, tsLength);
                    Nd4j.getExecutioner().exec(new BernoulliDistribution(lMask1, 0.5));
                    Nd4j.getExecutioner().exec(new BernoulliDistribution(lMask2, 0.5));
                }

                List<DataSet> l = Arrays.asList(new DataSet(in1, out1), new DataSet(in2, out2));
                DataSetIterator iter = new ExistingDataSetIterator(l);

//                System.out.println("Eval net 1");
                org.nd4j.evaluation.IEvaluation[] e1 = net1.doEvaluation(iter, new org.nd4j.evaluation.classification.Evaluation(), new org.nd4j.evaluation.classification.ROCMultiClass(), new org.nd4j.evaluation.regression.RegressionEvaluation());
//                System.out.println("Eval net 2");
                org.nd4j.evaluation.IEvaluation[] e2 = net2.doEvaluation(iter, new org.nd4j.evaluation.classification.Evaluation(), new org.nd4j.evaluation.classification.ROCMultiClass(), new org.nd4j.evaluation.regression.RegressionEvaluation());

                assertEquals(e1[0], e2[0]);
                assertEquals(e1[1], e2[1]);
                assertEquals(e1[2], e2[2]);
            }
        }
    }

    @Test
    public void testEvalSplitting2(){
        List<List<Writable>> seqFeatures = new ArrayList<>();
        List<Writable> step = Arrays.<Writable>asList(new FloatWritable(0), new FloatWritable(0), new FloatWritable(0));
        for( int i=0; i<30; i++ ){
            seqFeatures.add(step);
        }
        List<List<Writable>> seqLabels = Collections.singletonList(Collections.<Writable>singletonList(new FloatWritable(0)));

        SequenceRecordReader fsr = new CollectionSequenceRecordReader(Collections.singletonList(seqFeatures));
        SequenceRecordReader lsr = new CollectionSequenceRecordReader(Collections.singletonList(seqLabels));


        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(fsr, lsr, 1, -1, true,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123)
                .list()
                .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(3).nOut(3).build())
                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SIGMOID).lossFunction(LossFunctions.LossFunction.XENT)
                        .nIn(3).nOut(1).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(10).tBPTTBackwardLength(10)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.evaluate(testData);
    }

    @Test
    public void testEvaluativeListenerSimple(){
        //Sanity check: https://github.com/deeplearning4j/deeplearning4j/issues/5351

        // Network config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).seed(42)
                .updater(new Sgd(1e-6)).list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(2).activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                        LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3).weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).build())
                .build();

        // Instantiate model
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // Train-test split
        DataSetIterator iter = new IrisDataSetIterator(30, 150);
        DataSetIterator iterTest = new IrisDataSetIterator(30, 150);

        net.setListeners(new EvaluativeListener(iterTest, 3));

        for( int i=0; i<3; i++ ){
            net.fit(iter);
        }
    }

    @Test
    public void testMultiOutputEvalSimple(){
        Nd4j.getRandom().setSeed(12345);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .graphBuilder()
                .addInputs("in")
                .addLayer("out1", new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX).build(), "in")
                .addLayer("out2", new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX).build(), "in")
                .setOutputs("out1", "out2")
                .build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        List<MultiDataSet> list = new ArrayList<>();
        DataSetIterator iter = new IrisDataSetIterator(30, 150);
        while(iter.hasNext()){
            DataSet ds = iter.next();
            list.add(new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{ds.getFeatures()}, new INDArray[]{ds.getLabels(), ds.getLabels()}));
        }

        org.nd4j.evaluation.classification.Evaluation e = new org.nd4j.evaluation.classification.Evaluation();
        org.nd4j.evaluation.regression.RegressionEvaluation e2 = new org.nd4j.evaluation.regression.RegressionEvaluation();
        Map<Integer,org.nd4j.evaluation.IEvaluation[]> evals = new HashMap<>();
        evals.put(0, new org.nd4j.evaluation.IEvaluation[]{e});
        evals.put(1, new org.nd4j.evaluation.IEvaluation[]{e2});

        cg.evaluate(new IteratorMultiDataSetIterator(list.iterator(), 30), evals);

        assertEquals(150, e.getNumRowCounter());
        assertEquals(150, e2.getExampleCountPerColumn().getInt(0));
    }

    @Test
    public void testMultiOutputEvalCG(){
        //Simple sanity check on evaluation

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("0", new EmbeddingSequenceLayer.Builder().nIn(10).nOut(10).build(), "in")
                .layer("1", new LSTM.Builder().nIn(10).nOut(10).build(), "0")
                .layer("2", new LSTM.Builder().nIn(10).nOut(10).build(), "0")
                .layer("out1", new RnnOutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).build(), "1")
                .layer("out2", new RnnOutputLayer.Builder().nIn(10).nOut(20).activation(Activation.SOFTMAX).build(), "2")
                .setOutputs("out1", "out2")
                .build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        org.nd4j.linalg.dataset.MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{Nd4j.create(10, 1, 10)},
                new INDArray[]{Nd4j.create(10, 10, 10), Nd4j.create(10, 20, 10)});

        Map<Integer,org.nd4j.evaluation.IEvaluation[]> m = new HashMap<>();
        m.put(0, new org.nd4j.evaluation.IEvaluation[]{new org.nd4j.evaluation.classification.Evaluation()});
        m.put(1, new org.nd4j.evaluation.IEvaluation[]{new org.nd4j.evaluation.classification.Evaluation()});

        cg.evaluate(new SingletonMultiDataSetIterator(mds), m);
    }

    @Test
    public void testInvalidEvaluation(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(10).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(3).lossFunction(LossFunctions.LossFunction.MSE).activation(Activation.RELU).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        try {
            net.evaluate(iter);
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("Classifier") && e.getMessage().contains("Evaluation"));
        }

        try {
            net.evaluateROC(iter, 0);
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("Classifier") && e.getMessage().contains("ROC"));
        }

        try {
            net.evaluateROCMultiClass(iter, 0);
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("Classifier") && e.getMessage().contains("ROCMultiClass"));
        }

        ComputationGraph cg = net.toComputationGraph();
        try {
            cg.evaluate(iter);
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("Classifier") && e.getMessage().contains("Evaluation"));
        }

        try {
            cg.evaluateROC(iter, 0);
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("Classifier") && e.getMessage().contains("ROC"));
        }

        try {
            cg.evaluateROCMultiClass(iter, 0);
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("Classifier") && e.getMessage().contains("ROCMultiClass"));
        }


        //Disable validation, and check same thing:
        net.getLayerWiseConfigurations().setValidateOutputLayerConfig(false);
        net.evaluate(iter);
        net.evaluateROCMultiClass(iter, 0);

        cg.getConfiguration().setValidateOutputLayerConfig(false);
        cg.evaluate(iter);
        cg.evaluateROCMultiClass(iter, 0);
    }
}

/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.eval;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.*;

/**
 * Created by agibsonccc on 12/22/14.
 */
public class EvalTest {


    @Test
    public void testEval() {
        int classNum = 5;
        Evaluation eval = new Evaluation(classNum);

        // Testing the edge case when some classes do not have true positive
        INDArray trueOutcome = FeatureUtil.toOutcomeVector(0, 5);
        INDArray predictedOutcome = FeatureUtil.toOutcomeVector(0, 5);
        eval.eval(trueOutcome, predictedOutcome);
        assertEquals(1, eval.classCount(0));
        assertEquals(1.0, eval.f1(), 1e-1);

        // Testing more than one sample. eval() does not reset the Evaluation instance
        INDArray trueOutcome2 = FeatureUtil.toOutcomeVector(1, 5);
        INDArray predictedOutcome2 = FeatureUtil.toOutcomeVector(0, 5);
        eval.eval(trueOutcome2, predictedOutcome2);
        // Verified with sklearn in Python
        // from sklearn.metrics import classification_report
        // classification_report(['a', 'a'], ['a', 'b'], labels=['a', 'b', 'c', 'd', 'e'])
        assertEquals(eval.f1(), 0.6, 1e-1);
        // The first entry is 0 label
        assertEquals(1, eval.classCount(0));
        // The first entry is 1 label
        assertEquals(1, eval.classCount(1));
        // Two positives since two entries
        assertEquals(2, eval.positive(), 0);
        // The rest are negative
        assertEquals(8, eval.negative(), 0);
        // 2 rows and only the first is correct
        assertEquals(eval.accuracy(), 0.5, 0);
    }

    @Test
    public void testStringListLabels() {
        INDArray trueOutcome = FeatureUtil.toOutcomeVector(0, 2);
        INDArray predictedOutcome = FeatureUtil.toOutcomeVector(0, 2);

        List<String> labelsList = new ArrayList<>();
        labelsList.add("hobbs");
        labelsList.add("cal");

        Evaluation eval = new Evaluation(labelsList);

        eval.eval(trueOutcome, predictedOutcome);
        assertEquals(1, eval.classCount(0));
        assertEquals(labelsList.get(0), eval.getClassLabel(0));

    }

    @Test
    public void testStringHashLabels() {
        INDArray trueOutcome = FeatureUtil.toOutcomeVector(0, 2);
        INDArray predictedOutcome = FeatureUtil.toOutcomeVector(0, 2);

        Map<Integer, String> labelsMap = new HashMap<>();
        labelsMap.put(10, "hobbs");
        labelsMap.put(25, "cal");

        Evaluation eval = new Evaluation(labelsMap);

        eval.eval(trueOutcome, predictedOutcome);
        assertEquals(1, eval.classCount(0));
        assertEquals(labelsMap.get(10), eval.getClassLabel(10));

    }


    @Test
    public void testIris() {

        // Network config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(1)
                .seed(42)
                .learningRate(1e-6)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(2)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(2).nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .build();

        // Instantiate model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));

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
        INDArray testFeature = test.getFeatureMatrix();
        INDArray testLabel = test.getLabels();

        // Fitting model
        model.fit(train);
        // Get predictions from test feature
        INDArray testPredictedLabel = model.output(testFeature);

        // Eval with class number
        Evaluation eval = new Evaluation(3); //// Specify class num here
        eval.eval(testLabel, testPredictedLabel);
        double eval1F1 = eval.f1();
        double eval1Acc = eval.accuracy();

        // Eval without class number
        Evaluation eval2 = new Evaluation(); //// No class num
        eval2.eval(testLabel, testPredictedLabel);
        double eval2F1 = eval2.f1();
        double eval2Acc = eval2.accuracy();

        //Assert the two implementations give same f1 and accuracy (since one batch)
        assertTrue(eval1F1 == eval2F1 && eval1Acc == eval2Acc);

        Evaluation evalViaMethod = model.evaluate(new ListDataSetIterator(Collections.singletonList(test)));
        checkEvaluationEquality(eval,evalViaMethod);
    }

    @Test
    public void testEvalMasking(){
        int miniBatch = 5;
        int nOut = 3;
        int tsLength = 6;

        INDArray labels = Nd4j.zeros(miniBatch, nOut, tsLength);
        INDArray predicted = Nd4j.zeros(miniBatch, nOut, tsLength);

        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        for( int i=0; i<miniBatch; i++ ){
            for( int j=0; j<tsLength; j++ ){
                INDArray rand = Nd4j.rand(1,nOut);
                rand.divi(rand.sumNumber());
                predicted.put(new INDArrayIndex[]{NDArrayIndex.point(i),NDArrayIndex.all(),NDArrayIndex.point(j)},rand);
                int idx = r.nextInt(nOut);
                labels.putScalar(new int[]{i,idx,j},1.0);
            }
        }

        //Create a longer labels/predicted with mask for first and last time step
        //Expect masked evaluation to be identical to original evaluation
        INDArray labels2 = Nd4j.zeros(miniBatch, nOut, tsLength + 2);
        labels2.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(1,tsLength+1)},labels);
        INDArray predicted2 = Nd4j.zeros(miniBatch, nOut, tsLength + 2);
        predicted2.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, tsLength + 1)}, predicted);

        INDArray labelsMask = Nd4j.ones(miniBatch,tsLength+2);
        for( int i=0; i<miniBatch; i++ ){
            labelsMask.putScalar(new int[]{i,0},0.0);
            labelsMask.putScalar(new int[]{i,tsLength+1},0.0);
        }

        Evaluation evaluation = new Evaluation();
        evaluation.evalTimeSeries(labels,predicted);

        Evaluation evaluation2 = new Evaluation();
        evaluation2.evalTimeSeries(labels2,predicted2,labelsMask);

        System.out.println(evaluation.stats());
        System.out.println(evaluation2.stats());

        assertEquals(evaluation.accuracy(), evaluation2.accuracy(), 1e-12);
        assertEquals(evaluation.f1(), evaluation2.f1(), 1e-12);
        assertEquals(evaluation.falsePositives(), evaluation2.falsePositives(), 1e-12);
        assertEquals(evaluation.falseNegatives(),evaluation2.falseNegatives(),1e-12);
        assertEquals(evaluation.truePositives(),evaluation2.truePositives(),1e-12);
        assertEquals(evaluation.trueNegatives(), evaluation2.trueNegatives(), 1e-12);
        for( int i=0; i<nOut; i++) assertEquals(evaluation.classCount(i),evaluation2.classCount(i));
    }

    @Test
    public void testFalsePerfectRecall() {
        int testSize = 100;
        int numClasses = 5;
        int winner = 1;
        int seed = 241;

        INDArray labels = Nd4j.zeros(testSize, numClasses);
        INDArray predicted = Nd4j.zeros(testSize, numClasses);

        Nd4j.getRandom().setSeed(seed);
        Random r = new Random(seed);

        //Modelling the situation when system predicts the same class every time
        for(int i = 0; i < testSize; i++) {
            //Generating random prediction but with a guaranteed winner
            INDArray rand = Nd4j.rand(1, numClasses);
            rand.put(0, winner, rand.sumNumber());
            rand.divi(rand.sumNumber());
            predicted.put(new INDArrayIndex[]{NDArrayIndex.point(i),NDArrayIndex.all()}, rand);
            //Generating random label
            int label = r.nextInt(numClasses);
            labels.putScalar(new int[]{i,label},1.0);
        }

        //Explicitly specify the amount of classes
        Evaluation eval = new Evaluation(numClasses);
        eval.eval(labels, predicted);

        //For sure we shouldn't arrive at 100% recall unless we guessed everything right for every class
        assertNotEquals(1.0, eval.recall());
    }

    @Test
    public void testEvaluationMerging(){

        int nRows = 20;
        int nCols = 3;

        Random r = new Random(12345);
        INDArray actual = Nd4j.create(nRows,nCols);
        INDArray predicted = Nd4j.create(nRows, nCols);
        for( int i=0; i<nRows; i++ ){
            int x1 = r.nextInt(nCols);
            int x2 = r.nextInt(nCols);
            actual.putScalar(new int[]{i,x1},1.0);
            predicted.putScalar(new int[]{i,x2},1.0);
        }

        Evaluation evalExpected = new Evaluation();
        evalExpected.eval(actual,predicted);


        //Now: split into 3 separate evaluation objects -> expect identical values after merging
        Evaluation eval1 = new Evaluation();
        eval1.eval(actual.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()), predicted.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));

        Evaluation eval2 = new Evaluation();
        eval2.eval(actual.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()), predicted.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()));

        Evaluation eval3 = new Evaluation();
        eval3.eval(actual.get(NDArrayIndex.interval(10, nRows), NDArrayIndex.all()), predicted.get(NDArrayIndex.interval(10, nRows), NDArrayIndex.all()));

        eval1.merge(eval2);
        eval1.merge(eval3);

        checkEvaluationEquality(evalExpected,eval1);


        //Next: check evaluation merging with empty, and empty merging with non-empty
        eval1 = new Evaluation();
        eval1.eval(actual.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()), predicted.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));

        Evaluation evalInitiallyEmpty = new Evaluation();
        evalInitiallyEmpty.merge(eval1);
        evalInitiallyEmpty.merge(eval2);
        evalInitiallyEmpty.merge(eval3);
        checkEvaluationEquality(evalExpected, evalInitiallyEmpty);

        eval1.merge(new Evaluation());
        eval1.merge(eval2);
        eval1.merge(new Evaluation());
        eval1.merge(eval3);
        checkEvaluationEquality(evalExpected, eval1);
    }

    private static void checkEvaluationEquality(Evaluation evalExpected, Evaluation evalActual){
        assertEquals(evalExpected.accuracy(), evalActual.accuracy(), 1e-3);
        assertEquals(evalExpected.f1(), evalActual.f1(), 1e-3);
        assertEquals(evalExpected.getNumRowCounter(),evalActual.getNumRowCounter(), 1e-3);
        assertEquals(evalExpected.falseNegatives(),evalActual.falseNegatives(),1e-3);
        assertEquals(evalExpected.falsePositives(),evalActual.falsePositives(),1e-3);
        assertEquals(evalExpected.trueNegatives(),evalActual.trueNegatives(),1e-3);
        assertEquals(evalExpected.truePositives(),evalActual.truePositives(),1e-3);
        assertEquals(evalExpected.precision(),evalActual.precision(),1e-3);
        assertEquals(evalExpected.recall(),evalActual.recall(),1e-3);
        assertEquals(evalExpected.getConfusionMatrix(), evalActual.getConfusionMatrix());
    }
}

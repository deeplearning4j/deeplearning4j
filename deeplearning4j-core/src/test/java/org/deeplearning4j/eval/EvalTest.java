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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

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
    public void testStringLabels() {
        INDArray trueOutcome = FeatureUtil.toOutcomeVector(0, 2);
        INDArray predictedOutcome = FeatureUtil.toOutcomeVector(0, 2);

        List<String> labels = new ArrayList<>();
        labels.add("hobbs");
        labels.add("cal");

        Evaluation eval = new Evaluation(labels);

        eval.eval(trueOutcome, predictedOutcome);
        assertEquals(1, eval.classCount(0));
        assertEquals(labels.get(0), eval.getClassLabel(0));

    }

    @Test
    public void testIris() {

        // Network config
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()

                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(500)
                .seed(42)
                .learningRate(1e-1)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(4).nOut(3).weightInit(WeightInit.XAVIER).activation("softmax").build())
                .build();

        // Instantiate model
        OutputLayer l = LayerFactories.getFactory(conf.getLayer())
                .create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(500)), 0);

        // Train-test split
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        DataSet next = iter.next();
        next.shuffle();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110, new Random(42));
        // Train
        DataSet train = trainTest.getTrain();
        train.normalizeZeroMeanZeroUnitVariance();
        // Test
        DataSet test = trainTest.getTest();
        test.normalizeZeroMeanZeroUnitVariance();
        INDArray testFeature = test.getFeatureMatrix();
        INDArray testLabel = test.getLabels();

        // Fitting model
        l.fit(train);
        // Get predictions from test feature
        INDArray testPredictedLabel = l.output(testFeature);


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

        System.out.println(eval2F1);
        System.out.println(eval2Acc);

        //Assert the two implementations give same f1 and accuracy (since one batch)
        assertTrue(eval1F1 == eval2F1 && eval1Acc == eval2Acc);
    }
}

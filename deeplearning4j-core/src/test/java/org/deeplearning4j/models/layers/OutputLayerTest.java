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

package org.deeplearning4j.models.layers;

import java.util.Arrays;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.stepfunctions.GradientStepFunction;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/1/14.
 */
public class OutputLayerTest {
    private static final Logger log = LoggerFactory.getLogger(OutputLayerTest.class);


    @Test
    public void testIris() {
        LayerFactory layerFactory = LayerFactories.getFactory(OutputLayer.class);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT).optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .activationFunction("softmax")
                .iterations(100).weightInit(WeightInit.ZERO)
                .learningRate(1e-1).nIn(4).nOut(3).layerFactory(layerFactory).build();

        OutputLayer l = layerFactory.create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)));
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        DataSet next = iter.next();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        trainTest.getTrain().normalizeZeroMeanZeroUnitVariance();
        l.fit(trainTest.getTrain());


        DataSet test = trainTest.getTest();
        test.normalizeZeroMeanZeroUnitVariance();
        Evaluation eval = new Evaluation();
        INDArray output = l.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(),output);
        log.info("Score " +eval.stats());


    }


}

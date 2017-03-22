/*-
 *  * Copyright 2016 Skymind,Inc.
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
 */

package org.deeplearning4j.arbiter.scoring;


import org.deeplearning4j.arbiter.scoring.graph.*;
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetAccuracyScoreFunction;
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetF1ScoreFunction;
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetLossScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetRegressionScoreFunction;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * ScoreFunctions provides static methods for getting score functions for DL4J MultiLayerNetwork and ComputationGraph
 *
 * @author Alex Black
 */
public class ScoreFunctions {

    private ScoreFunctions() {
    }

    /**
     * Calculate the loss (score/loss function value) on a test set, for a MultiLayerNetwork
     *
     * @param average Average (divide by number of examples)
     */
    public static ScoreFunction<MultiLayerNetwork, Object> testSetLoss(boolean average) {
        return new TestSetLossScoreFunction(average);
    }

    /**
     * Calculate the loss (score/loss function value) on a MultiDataSetIterator (test set), for a (single output) ComputationGraph
     *
     * @param average Average (divide by number of examples)
     */
    public static ScoreFunction<ComputationGraph, Object> testSetLossGraph(boolean average) {
        return new GraphTestSetLossScoreFunction(average);
    }

    /**
     * Calculate the loss (score/loss function value) on a DataSetIterator (test set), for a (single output) ComputationGraph
     *
     * @param average Average (divide by number of examples)
     */
    public static ScoreFunction<ComputationGraph, Object> testSetLossGraphDataSet(boolean average) {
        return new GraphTestSetLossScoreFunctionDataSet(average);
    }

    /**
     * Calculate the accuracy on a test set, for a MultiLayerNetwork
     */
    public static ScoreFunction<MultiLayerNetwork, Object> testSetAccuracy() {
        return new TestSetAccuracyScoreFunction();
    }

    /**
     * Calculate the accuracy on a test set (MultiDataSetIterator) for a ComputationGraph
     */
    public static ScoreFunction<ComputationGraph, Object> testSetAccuracyGraph() {
        return new GraphTestSetAccuracyScoreFunction();
    }

    /**
     * Calculate the accuracy on a test set (DataSetIterator) for a ComputationGraph
     */
    public static ScoreFunction<ComputationGraph, Object> testSetAccuracyGraphDataSet() {
        return new GraphTestSetAccuracyScoreFunctionDataSet();
    }


    /**
     * Calculate the f1 score on a test set, for a MultiLayerNetwork
     */
    public static ScoreFunction<MultiLayerNetwork, Object> testSetF1() {
        return new TestSetF1ScoreFunction();
    }

    /**
     * Calculate the f1 score on a test set (MultiDataSetIterator), for a ComputationGraph
     */
    public static ScoreFunction<ComputationGraph, Object> testSetF1Graph() {
        return new GraphTestSetF1ScoreFunction();
    }

    /**
     * Calculate the f1 score on a test set (DataSetIterator), for a ComputationGraph
     */
    public static ScoreFunction<ComputationGraph, Object> testSetF1GraphDataSet() {
        return new GraphTestSetF1ScoreFunctionDataSet();
    }

    /**
     * Calculate a regression value (MSE, MAE etc) on a test set (DataSetIterator) for a MultiLayerNetwork
     */
    public static ScoreFunction<MultiLayerNetwork, Object> testSetRegression(RegressionValue regressionValue) {
        return new TestSetRegressionScoreFunction(regressionValue);
    }

    /**
     * Calculate a regression value (MSE, MAE etc) on a test set (MultiDataSetIterator) for a ComputationGraph
     */
    public static ScoreFunction<ComputationGraph, Object> testSetRegressionGraph(RegressionValue regressionValue) {
        return new GraphTestSetRegressionScoreFunction(regressionValue);
    }

    /**
     * Calculate a regression value (MSE, MAE etc) on a test set (DataSetIterator) for a MultiLayerNetwork
     */
    public static ScoreFunction<ComputationGraph, Object> testSetRegressionGraphDataSet(RegressionValue regressionValue) {
        return new GraphTestSetRegressionScoreFunctionDataSet(regressionValue);
    }

}

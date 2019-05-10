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

package org.deeplearning4j.arbiter.scoring;


import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.impl.TestSetAccuracyScoreFunction;
import org.deeplearning4j.arbiter.scoring.impl.TestSetF1ScoreFunction;
import org.deeplearning4j.arbiter.scoring.impl.TestSetLossScoreFunction;
import org.deeplearning4j.arbiter.scoring.impl.TestSetRegressionScoreFunction;

/**
 * ScoreFunctions provides static methods for getting score functions for DL4J MultiLayerNetwork and ComputationGraph
 *
 * @author Alex Black
 */
public class ScoreFunctions {

    private ScoreFunctions() {}

    /**
     * Calculate the loss (score/loss function value) on a test set, for a MultiLayerNetwork
     *
     * @param average Average (divide by number of examples)
     */
    public static ScoreFunction testSetLoss(boolean average) {
        return new TestSetLossScoreFunction(average);
    }

    /**
     * Calculate the accuracy on a test set, for a MultiLayerNetwork
     */
    public static ScoreFunction testSetAccuracy() {
        return new TestSetAccuracyScoreFunction();
    }


    /**
     * Calculate the f1 score on a test set
     */
    public static ScoreFunction testSetF1() {
        return new TestSetF1ScoreFunction();
    }

    /**
     * Calculate a regression value (MSE, MAE etc) on a test set
     */
    public static ScoreFunction testSetRegression(RegressionValue regressionValue) {
        return new TestSetRegressionScoreFunction(regressionValue);
    }

}
